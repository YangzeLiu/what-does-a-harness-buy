#!/usr/bin/env python3
"""Replace the silent `curl mirror/install.sh | sh` uv line in task Dockerfiles with a verified install.

Why (2026-09-08): with the host mirror unreachable, `curl -sSf ... | sh` fed sh an empty script,
the RUN exited 0, and Docker cached a uv-less layer that every later trial of that task reused.
The verifier's `uv run parser.py` then failed ("uv: command not found") for CC/OpenCode trials.
The new line tries the mirror, then astral.sh, then pip, and ends with `uv --version` so a build
without uv fails loudly. Changing the line also changes the layer cache key -> clean rebuild.

usage: patch_dockerfile_uv.py <tasks_root> [--apply]   (default: dry run; idempotent; backup Dockerfile.orig-uvfix)
"""
import sys, glob, os, shutil
root = sys.argv[1]; apply = "--apply" in sys.argv
OLD = "RUN curl -sSf http://172.17.0.1:8090/install.sh | INSTALLER_DOWNLOAD_URL=http://172.17.0.1:8090 sh\n"
MARK = "# agent-harness 2026-09-08: verified uv install"
NEW = ("RUN ( curl -fsS --connect-timeout 10 --max-time 120 --retry 3 http://172.17.0.1:8090/install.sh -o /tmp/uv-install.sh "
       "&& INSTALLER_DOWNLOAD_URL=http://172.17.0.1:8090 sh /tmp/uv-install.sh ) "
       "|| ( curl -LsSf --connect-timeout 20 --max-time 600 --retry 5 --retry-all-errors --retry-delay 10 https://astral.sh/uv/0.7.13/install.sh | sh ) "
       "|| python -m pip install -q uv==0.7.13 ; /root/.local/bin/uv --version || uv --version\n"
       + MARK + " (mirror, then astral.sh, then pip; the build fails if uv is still missing).\n"
       "#   The previous `curl -sSf .../install.sh | ... sh` built uv-less images silently whenever the mirror was down.\n")
files = sorted(glob.glob(os.path.join(root, "*", "environment", "Dockerfile")))
todo = done = odd = 0
for f in files:
    s = open(f).read()
    if MARK in s: done += 1; continue
    if s.count(OLD) != 1: odd += 1; print("ODD:", f); continue
    todo += 1
    if apply:
        bak = f + ".orig-uvfix"
        if not os.path.exists(bak): shutil.copy2(f, bak)
        open(f, "w").write(s.replace(OLD, NEW))
print(f"dockerfiles {len(files)} already-patched {done} {'patched' if apply else 'would-patch'} {todo} odd {odd}")
