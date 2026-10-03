#!/usr/bin/env python3
"""Make every pool task's verifier self-healing when `uv` is absent from the image.

Root cause (2026-09-08): tasks_pool/*/environment/Dockerfile installs uv with
`curl -sSf http://172.17.0.1:8090/install.sh | ... sh`; when the host mirror is unreachable the
pipe exits 0 with empty input, so the image builds *without* uv and the verifier's
`uv run parser.py` dies with "uv: command not found" (no report.json -> reward 0).
mini-swe-agent's Harbor installer reinstalls uv itself, which is why only CC/OC trials were hit.

usage: patch_testsh_uv.py <tasks_root> [--apply]   (default: dry run)
Idempotent: skips files that already carry the marker. Originals saved as test.sh.orig-uvfix.
"""
import sys, glob, os, shutil
root = sys.argv[1]; apply = "--apply" in sys.argv
MARK = "# agent-harness 2026-09-08 uv self-heal"
ANCHOR_PREV = 'export PATH="/root/.local/bin:$PATH"\n'
ANCHOR = 'uv run parser.py | tee -a "$LOG_FILE"\n'
BLOCK = MARK + ''' (image may lack uv when its build-time mirror fetch failed silently)
if ! command -v uv >/dev/null 2>&1; then
  echo "[verifier] uv missing in image; installing"
  ( curl -sSf --connect-timeout 10 --max-time 120 http://172.17.0.1:8090/install.sh -o /tmp/uv-install.sh \\
      && INSTALLER_DOWNLOAD_URL=http://172.17.0.1:8090 UV_INSTALLER_GITHUB_BASE_URL=http://172.17.0.1:8090 sh /tmp/uv-install.sh ) \\
    || curl -LsSf --connect-timeout 20 --max-time 600 --retry 5 --retry-all-errors --retry-delay 10 https://astral.sh/uv/0.7.13/install.sh | sh \\
    || python -m pip install -q uv
  export PATH="/root/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 && echo "[verifier] uv now available: $(uv --version)" || echo "[verifier] uv install FAILED"
fi
'''
files = sorted(glob.glob(os.path.join(root, "*", "tests", "test.sh")))
todo = done = odd = 0
for f in files:
    s = open(f).read()
    if MARK in s: done += 1; continue
    if s.count(ANCHOR) != 1 or (ANCHOR_PREV + ANCHOR) not in s: odd += 1; print("ODD layout:", f); continue
    todo += 1
    if apply:
        bak = f + ".orig-uvfix"
        if not os.path.exists(bak): shutil.copy2(f, bak)
        open(f, "w").write(s.replace(ANCHOR_PREV + ANCHOR, ANCHOR_PREV + BLOCK + ANCHOR))
print(f"files {len(files)} already-patched {done} {'patched' if apply else 'would-patch'} {todo} odd {odd}")
