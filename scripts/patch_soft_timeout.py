"""Harness-local patch: HARBOR_AGENT_SOFT_TIMEOUT_SEC wraps the OpenCode / Claude Code binary in
coreutils `timeout -s TERM -k 30 <sec>` so the agent is gone before Harbor's own agent timeout
starts the verifier (2026-09-08 audit: agents kept editing for up to 26 min during verification).
Exit 124 leaves /logs/agent/soft-timeout.txt; env unset -> command byte-identical to before.
Usage: python3 patch_soft_timeout.py <installed-agents-dir>   (idempotent, keeps .pre-soft backups)
"""
import os, py_compile, shutil, sys

HELPER = '''

def _soft_timeout_prefix():
    """Harness-local (2026-09-08): HARBOR_AGENT_SOFT_TIMEOUT_SEC>0 wraps the agent binary in
    coreutils timeout so it has exited before Harbor's agent timeout hands over to the verifier."""
    try:
        sec = int(os.environ.get("HARBOR_AGENT_SOFT_TIMEOUT_SEC", "0") or 0)
    except ValueError:
        sec = 0
    return f"timeout -s TERM -k 30 {sec} " if sec > 0 else ""


def _soft_timeout_marker(rc_expr, logs_dir="/logs/agent"):
    if not _soft_timeout_prefix():
        return ""
    return (f'_rc={rc_expr}; if [ "$_rc" -eq 124 ]; then echo "soft timeout reached" '
            f'> {logs_dir}/soft-timeout.txt; exit 124; fi; ')
'''

def patch(path, edits):
    s = open(path).read()
    if "_soft_timeout_prefix" in s:
        print(f"{path}: already patched"); return
    shutil.copy2(path, path + ".pre-soft")
    if "\nimport os\n" not in s:
        s = s.replace("\nimport json\n", "\nimport json\nimport os\n", 1)
        assert "\nimport os\n" in s, "could not add import os"
    for old, new in edits:
        n = s.count(old)
        assert n == 1, f"{path}: pattern count {n} for {old[:70]!r}"
        s = s.replace(old, new)
    # helper after the imports: before the first top-level def/class
    idx = min(i for i in (s.find("\nclass "), s.find("\ndef ")) if i > 0)
    s = s[:idx] + HELPER + s[idx:]
    open(path, "w").write(s)
    py_compile.compile(path, doraise=True)
    print(f"{path}: patched ({len(edits)} edits)")

d = sys.argv[1]
oc = os.path.join(d, "opencode.py")
s = open(oc).read()
edits = [('f"opencode --model={self.model_name} run --format=json "',
          'f"{_soft_timeout_prefix()}opencode --model={self.model_name} run --format=json "')]
if "for _try in 1 2 3 4" in s:      # host-a relaunch-loop variant: loop already exits with opencode's rc
    edits.append(('\'| tee -a /logs/agent/opencode-retries.txt >&2; sleep 5; continue; fi; break; done; exit "$_rc"\'',
                  '\'| tee -a /logs/agent/opencode-retries.txt >&2; sleep 5; continue; fi; break; done; \'\n'
                  '                f\'{_soft_timeout_marker("$_rc")}exit "$_rc"\''))
else:                                # plain pipeline variant (local, host-b)
    edits.append(('f"2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt"',
                  'f"2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt; "\n'
                  '                f\'{_soft_timeout_marker("${PIPESTATUS[0]}")}\''))
patch(oc, edits)

cc = os.path.join(d, "claude_code.py")
patch(cc, [
    ('f"claude --verbose --output-format=stream-json "',
     'f"{_soft_timeout_prefix()}claude --verbose --output-format=stream-json "'),
    ('''                    f"--print 2>&1 | tee "
                    f"{(self.environment_logs_dir / 'claude-code.txt').as_posix()}"
                ),''',
     '''                    f"--print 2>&1 | tee "
                    f"{(self.environment_logs_dir / 'claude-code.txt').as_posix()}; "
                    f'{_soft_timeout_marker("${PIPESTATUS[1]}", self.environment_logs_dir.as_posix())}'
                ),'''),
])
