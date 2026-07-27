#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
OrchestrAI pipeline trigger
==========================

For each batch (one per platform/device[+extra-tags]) builds a pipeline plan —
one group per playbook, sharing the batch's hardware — and POSTs it to the
OrchestrAI pipeline job (buildWithParameters). Polls the queue item until the
build starts and records its URL so the matrix job can wait on it.

Environment-specific values (OrchestrAI pipeline URL/job, provisioning scripts, TheRock
URLs, Linux/Windows driver sources, run settings) come from
.github/orchestrai-config.yml. Credentials come from env
(ORCHESTRAI_PIPELINE_USER / ORCHESTRAI_PIPELINE_TOKEN).

Usage:
    orchestrai_trigger.py [--config .github/orchestrai-config.yml] [--dry-run]

Inputs (env):
    BATCHES_JSON            batches from orchestrai_matrix.py
    GIT_REF                 github.ref (recorded as PLAYBOOK_REF)
    MACHINES_PER_HW_GROUP   override; falls back to config default
    ORCHESTRAI_PIPELINE_USER, ORCHESTRAI_PIPELINE_TOKEN

Outputs:
    build_urls={...}  -> $GITHUB_OUTPUT   (batch_id -> OrchestrAI pipeline build URL)
    --dry-run prints each plan/builds payload and POSTs nothing.

Fails fast (exit 1) on a misconfigured run rather than acquiring scarce
hardware that can't provision: missing config keys, missing creds, or a batch
whose required provisioning value (TheRock URL or Linux/Windows driver source)
is unset.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

# curl timeouts so a stalled OrchestrAI pipeline can never hang the job (vs the default: wait
# forever). Values are generous; the point is a finite ceiling, not tight SLAs.
POST_TIMEOUT = ["--connect-timeout", "15", "--max-time", "120"]
POLL_TIMEOUT = ["--connect-timeout", "10", "--max-time", "30"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_env_overrides(cfg):
    """Internal coordinates are kept out of the committed (public) config and
    supplied at run time via repo variables. Env values win over config."""
    j = cfg.setdefault("pipeline", {})
    j["url"] = os.environ.get("ORCHESTRAI_PIPELINE_URL") or j.get("url", "")
    j["job"] = os.environ.get("ORCHESTRAI_PIPELINE_JOB") or j.get("job", "")
    prov = cfg.setdefault("provisioning", {})
    therock = os.environ.get("ORCHESTRAI_THEROCK_URL")
    if therock:
        prov["therock_url"] = therock
    linux_driver = os.environ.get("ORCHESTRAI_LINUX_DRIVER_SOURCE")
    if linux_driver:
        prov.setdefault("linux_kernel_driver", {})["source"] = linux_driver
    npu_xrt = os.environ.get("ORCHESTRAI_RYZENAI_NPU_XRT_URL")
    if npu_xrt:
        prov["ryzenai_npu_xrt_url"] = npu_xrt
    driver = os.environ.get("ORCHESTRAI_WINDOWS_DRIVER_SOURCE")
    if driver:
        prov.setdefault("windows_driver", {})["source"] = driver
    return cfg


def validate_config(cfg, batches):
    """Return a list of missing required config keys (given the batches' platforms)."""
    errs = []
    for k in ("test_path_template", "playbook_repo"):
        if not cfg.get(k):
            errs.append(k)
    rs = cfg.get("run_settings") or {}
    for k in ("max_duration", "max_test_case_duration", "acquire_timeout"):
        if k not in rs:
            errs.append(f"run_settings.{k}")
    prov = cfg.get("provisioning") or {}
    platforms = {b.get("platform") for b in batches.values()}
    if "linux" in platforms and not prov.get("linux_install_scripts"):
        errs.append("provisioning.linux_install_scripts")
    if "windows" in platforms and not prov.get("windows_install_scripts"):
        errs.append("provisioning.windows_install_scripts")
    return errs


def make_plan(batch, git_ref, cfg, machines_per_hw_group, repo, sha, rocm_index_url,
              hf_token=""):
    platform = batch["platform"]
    device = batch["arch"]
    tags = batch["tags"]
    test = cfg["test_path_template"].format(platform=platform)
    rs = cfg["run_settings"]

    groups = []
    for pb_id in batch["playbooks"]:
        variables = {
            "PLAYBOOK_REPO": repo,
            "PLAYBOOK_SHA": sha,
            "PLAYBOOK_REF": git_ref,
            "PLAYBOOK_ID": pb_id,
            "PLAYBOOK_PLATFORM": platform,
            "PLAYBOOK_DEVICE": device,
            "ROCM_MULTI_ARCH_INDEX_URL": rocm_index_url,
        }
        # Authenticate HuggingFace traffic (models/datasets primed by deps fall
        # back to a live HF pull on a mirror miss; unauthenticated calls from the
        # shared egress IP hit HF's per-IP rate limit). Optional: omitted when the
        # secret is unset so runs still work (just unauthenticated).
        if hf_token:
            variables["HF_TOKEN"] = hf_token
        groups.append({
            "id": f"playbook-{pb_id}",
            "level": "L4-sys",
            "tests": [test],
            "maas_tags": tags,
            "variables": variables,
        })

    return {
        "source": "external-playbook-batch",
        "groups": groups,
        "run_settings": {
            "max_duration": rs["max_duration"],
            "max_test_case_duration": rs["max_test_case_duration"],
            "acquire_timeout": rs["acquire_timeout"],
            "machines_per_hw_group": machines_per_hw_group,
        },
    }


def make_builds(batch, cfg):
    """Return (builds, missing) — missing names required provisioning vars left empty."""
    platform = batch["platform"]
    prov = cfg.get("provisioning", {})
    missing = []

    if platform == "windows":
        # Copy so appending per-playbook extras below doesn't mutate cfg.
        scripts = list(prov.get("windows_install_scripts", []))
        drv = prov.get("windows_driver", {})
        source = drv.get("source", "")
        if not source:
            missing.append("ORCHESTRAI_WINDOWS_DRIVER_SOURCE")
        build_vars = {"driver_source": source, "driver_copy": drv.get("copy", "direct")}
    else:
        scripts = list(prov.get("linux_install_scripts", []))
        device = batch.get("arch", "")
        device_family = (cfg.get("device_families") or {}).get(device)
        if device_family not in {"radeon", "ryzen_apu"}:
            missing.append(f"device_families.{device}")
        kernel_driver = prov.get("linux_kernel_driver", {})
        if device_family == "radeon":
            source = kernel_driver.get("source", "")
            if not source:
                missing.append("ORCHESTRAI_LINUX_DRIVER_SOURCE")
            # The kernel driver must be active before TheRock user-space is
            # installed. Keep this device-specific so APUs retain the inbox
            # amdgpu module they are validated with.
            scripts = list(kernel_driver.get("install_scripts", [])) + scripts
        # TheRock ships a single multi-arch tarball, so the URL is used as-is
        # (no per-gfx templating).
        url = prov.get("therock_url", "")
        if not url:
            missing.append("ORCHESTRAI_THEROCK_URL")
        build_vars = {"THEROCK_URL": url}
        if device_family == "radeon":
            build_vars["driver_source"] = source
        if "cvml" in batch.get("playbooks", []):
            npu_xrt_url = prov.get("ryzenai_npu_xrt_url", "")
            if not npu_xrt_url:
                missing.append("ORCHESTRAI_RYZENAI_NPU_XRT_URL")
            build_vars["RAI_NPU_XRT_URL"] = npu_xrt_url

    # Per-playbook extra provisioning scripts (e.g. enabling WSL for
    # openclaw-lemonade-server). Appended only to batches that actually contain
    # the playbook, so the cost (an extra feature install + reboot) is scoped to
    # batches with that playbook instead of every run on the platform. This is
    # batch-level, not group-level: any other playbook co-scheduled in the same
    # batch also gets these scripts, so they must be idempotent. Each entry may
    # carry an optional "platforms" filter (e.g. windows-only); it is stripped
    # before the script list is handed to the pipeline, which expects only
    # {script, reboot_after}.
    extra_map = cfg.get("extra_install_scripts", {})
    seen = {s.get("script") for s in scripts}
    for pb_id in batch.get("playbooks", []):
        for entry in extra_map.get(pb_id, []):
            plats = entry.get("platforms")
            if plats and platform not in plats:
                continue
            script = entry.get("script")
            if not script or script in seen:
                continue
            seen.add(script)
            scripts.append({
                "script": script,
                "reboot_after": bool(entry.get("reboot_after", False)),
            })

    builds = {"install_scripts": scripts, "vars": build_vars}
    return builds, missing


def trigger(plan, builds, platform, pipeline, user, token):
    # Use a real temp dir (not hardcoded /tmp) so this also works on non-POSIX
    # runners; cleaned up in the finally below.
    tmpdir = tempfile.mkdtemp(prefix="oai-")
    plan_file = os.path.join(tmpdir, "plan.json")
    builds_file = os.path.join(tmpdir, "builds.json")
    headers_file = os.path.join(tmpdir, "headers.txt")
    try:
        with open(plan_file, "w") as f:
            json.dump(plan, f)
        with open(builds_file, "w") as f:
            json.dump(builds, f)

        os_image = "windows" if platform == "windows" else "ubuntu"
        r = subprocess.run(
            ["curl", "-sf", *POST_TIMEOUT, "-o", os.devnull, "-D", headers_file,
             "-u", f"{user}:{token}",
             "-X", "POST", f"{pipeline['url']}/job/{pipeline['job']}/buildWithParameters",
             "--data-urlencode", f"PLAN_JSON@{plan_file}",
             "--data-urlencode", f"BUILDS_JSON@{builds_file}",
             "--data-urlencode", f"OS_IMAGE={os_image}"],
            capture_output=True, text=True)
        if r.returncode != 0:
            return None

        queue_url = ""
        for line in open(headers_file).read().split("\n"):
            if line.lower().startswith("location:"):
                queue_url = line.split(":", 1)[1].strip()
                break
        if not queue_url:
            return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    for _ in range(60):
        time.sleep(10)
        try:
            q = subprocess.run(
                ["curl", "-sf", *POLL_TIMEOUT, "-u", f"{user}:{token}", f"{queue_url}api/json"],
                capture_output=True, text=True)
            if q.returncode != 0:
                continue
            exe = json.loads(q.stdout).get("executable", {})
            if exe and exe.get("url"):
                return exe["url"]
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=".github/orchestrai-config.yml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = apply_env_overrides(load_config(args.config))
    batches = json.loads(os.environ.get("BATCHES_JSON", "{}") or "{}")
    git_ref = os.environ.get("GIT_REF", "")
    user = os.environ.get("ORCHESTRAI_PIPELINE_USER", "")
    token = os.environ.get("ORCHESTRAI_PIPELINE_TOKEN", "")

    cfg_errs = validate_config(cfg, batches)
    if cfg_errs:
        print(f"::error::orchestrai-config.yml missing required keys: {', '.join(cfg_errs)}",
              file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        need = [f"pipeline.{k}" for k in ("url", "job") if not cfg["pipeline"].get(k)]
        if not user:
            need.append("ORCHESTRAI_PIPELINE_USER")
        if not token:
            need.append("ORCHESTRAI_PIPELINE_TOKEN")
        if need:
            print(f"::error::OrchestrAI not configured — missing: {', '.join(need)} "
                  f"(set ORCHESTRAI_PIPELINE_URL/JOB vars and ORCHESTRAI_PIPELINE_* secrets)",
                  file=sys.stderr)
            sys.exit(1)

    rs = cfg["run_settings"]
    try:
        mphg = int(os.environ.get("MACHINES_PER_HW_GROUP", "")
                   or rs.get("default_machines_per_hw_group", 1))
    except ValueError:
        mphg = 1

    # The pipeline clones PLAYBOOK_REPO@PLAYBOOK_SHA on the test machine. By
    # default that's the configured repo at main; the !orc PR path overrides both
    # so the PR's own code (including a fork head) is what actually gets tested.
    repo = os.environ.get("ORCHESTRAI_PLAYBOOK_REPO") or cfg["playbook_repo"]
    sha = os.environ.get("ORCHESTRAI_PLAYBOOK_SHA") or "main"
    rocm_index_url = os.environ.get("ORCHESTRAI_ROCM_MULTI_ARCH_INDEX_URL", "")
    if not rocm_index_url:
        print("::error::ORCHESTRAI_ROCM_MULTI_ARCH_INDEX_URL repository variable is required",
              file=sys.stderr)
        sys.exit(1)

    # Optional: HuggingFace token so live HF pulls (dep cache misses, uncached
    # datasets) are authenticated and avoid the shared-IP rate limit. Not
    # required -- runs still work unauthenticated, just exposed to 429s.
    hf_token = os.environ.get("ORCHESTRAI_HF_TOKEN", "")
    if not hf_token:
        print("::warning::ORCHESTRAI_HF_TOKEN not set — HuggingFace traffic will be "
              "unauthenticated and may hit the shared-IP rate limit (429)", file=sys.stderr)

    # Prepare every batch first, and fail fast (before any POST) if a batch is
    # missing required provisioning — otherwise we'd acquire a scarce machine
    # that can't install its GPU stack and only fail much later on hardware.
    prepared = []
    prov_missing = {}
    for bid, batch in batches.items():
        builds, missing = make_builds(batch, cfg)
        if missing:
            prov_missing[bid] = missing
        prepared.append((bid, batch,
                         make_plan(batch, git_ref, cfg, mphg, repo, sha,
                                   rocm_index_url, hf_token),
                         builds))

    if not args.dry_run and prov_missing:
        for bid, miss in prov_missing.items():
            print(f"::error::batch {bid}: unset provisioning {miss} — "
                  f"set the corresponding repo variable(s)", file=sys.stderr)
        sys.exit(1)

    build_urls = {}
    for bid, batch, plan, builds in prepared:
        print(f"Batch {bid}: {', '.join(batch['playbooks'])}", file=sys.stderr)
        if args.dry_run:
            print(f"--- plan for {bid} ---")
            print(json.dumps(plan, indent=2))
            print(f"--- builds for {bid} ---")
            print(json.dumps(builds, indent=2))
            continue
        url = trigger(plan, builds, batch["platform"], cfg["pipeline"], user, token)
        if url:
            print(f"  Build: {url}", file=sys.stderr)
            build_urls[bid] = url
        else:
            print(f"  WARNING: trigger failed / timed out for {bid}", file=sys.stderr)

    print(f"\nTriggered {len(build_urls)} OrchestrAI pipeline build(s)", file=sys.stderr)
    out = os.environ.get("GITHUB_OUTPUT")
    if out and not args.dry_run:
        with open(out, "a") as f:
            f.write(f"build_urls={json.dumps(build_urls)}\n")


if __name__ == "__main__":
    main()
