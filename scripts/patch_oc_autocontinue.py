"""Harness-local patch (2026-09-14): cutoff-recovery / "keep going" arm for the OpenCode agent.
HARBOR_OC_AUTOCONTINUE=N: after the stock `opencode run`, if the last step ended with finish_reason=length
(32k runaway think cut off) OR the repository is unmodified, send up to N follow-up turns with
`opencode run --continue "<MSG>"` (same session), appending JSON events to /logs/agent/opencode.txt and
logging rounds to /logs/agent/autocontinue.txt. MSG defaults to Claude Code's own post-max_tokens recovery
prompt (so the arm = "transplant CC's cutoff-recovery policy into OpenCode"); override with
HARBOR_OC_AUTOCONTINUE_MSG. Env unset -> command byte-identical to before.
Usage: python3 patch_oc_autocontinue.py <installed-agents-dir>   (idempotent; re-patches from the dated backup)
"""
import os, py_compile, shutil, sys

MSG = ("Output token limit hit. Resume directly — no apology, no recap of what you were doing. "
       "Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.")

HELPER = '''

def _autocontinue_msg():
    return os.environ.get("HARBOR_OC_AUTOCONTINUE_MSG") or _AUTOCONTINUE_MSG


def _autocontinue_rounds():
    """Harness-local (2026-09-14): HARBOR_OC_AUTOCONTINUE=N enables N follow-up turns when the stock run
    ends with a length-cut step or without modifying the repository (termination-policy ablation)."""
    try:
        return int(os.environ.get("HARBOR_OC_AUTOCONTINUE", "0") or 0)
    except ValueError:
        return 0


def _autocontinue_block(run_cmd, logs_dir="/logs/agent"):
    """Shell snippet inserted after the stock run; expects $_rc = stock run exit code.
    run_cmd is the `opencode ... run --format=json --continue ... -- <msg>` command (no redirection)."""
    n = _autocontinue_rounds()
    if n <= 0:
        return ""
    # repo = cwd if it is a git work tree, else /testbed; "modified" = tracked diff or untracked files
    # (SWE-bench images ignore __pycache__ etc.); if neither is a repo, treat as modified.
    mod = ('_gd=.; git -C . rev-parse --is-inside-work-tree >/dev/null 2>&1 || _gd=/testbed; '
           'if git -C "$_gd" rev-parse --is-inside-work-tree >/dev/null 2>&1; then _mod=0; '
           'git -C "$_gd" diff --quiet 2>/dev/null || _mod=1; '
           '[ -n "$(git -C \\"$_gd\\" ls-files --others --exclude-standard 2>/dev/null | head -1)" ] && _mod=1; '
           'else _mod=1; fi; ')
    # last step_finish event ended with finish_reason=length (runaway think cut off at the output cap)
    cut = ('_cut=0; grep \\'"type":"step_finish"\\' ' + logs_dir + '/opencode.txt 2>/dev/null | tail -1 '
           '| grep -q \\'"reason":"length"\\' && _cut=1; ')
    return (f'_ac_used=0; : > {logs_dir}/autocontinue.txt; '
            f'for _ac_k in $(seq 1 {n}); do {mod}{cut}'
            f'if {{ [ "$_mod" -eq 1 ] && [ "$_cut" -eq 0 ]; }} || [ "$_rc" -eq 124 ]; then '
            f'echo "round=$_ac_k modified_before=$_mod cut_before=$_cut rc_before=$_rc skipped" >> {logs_dir}/autocontinue.txt; break; fi; '
            f'_ac_used=$((_ac_used+1)); '
            f'{run_cmd} 2>&1 </dev/null | stdbuf -oL tee -a {logs_dir}/opencode.txt; '
            '_rc=${PIPESTATUS[0]}; '
            f'echo "round=$_ac_k modified_before=$_mod cut_before=$_cut rc=$_rc" >> {logs_dir}/autocontinue.txt; done; '
            f'{mod}{cut}echo "rounds_used=$_ac_used modified_final=$_mod cut_final=$_cut" >> {logs_dir}/autocontinue.txt; ')


def _autocontinue_plain(run_cmd, logs_dir="/logs/agent"):
    """Plain-pipeline variant (no relaunch loop): capture the stock rc first."""
    if _autocontinue_rounds() <= 0:
        return ""
    return '_rc=${PIPESTATUS[0]}; ' + _autocontinue_block(run_cmd, logs_dir)
'''

def patch(path):
    s = open(path).read()
    bak = path + ".orig-20260914-autocontinue"
    if "_autocontinue_block" in s:
        assert os.path.exists(bak), "already patched but backup missing"
        s = open(bak).read(); print(f"{path}: re-patching from {os.path.basename(bak)}")
    else:
        shutil.copy2(path, bak)
    assert "_soft_timeout_prefix" in s, "expects the soft-timeout patched variant"
    loop_old = ('                \'| tee -a /logs/agent/opencode-retries.txt >&2; sleep 5; continue; fi; break; done; \'\n'
                '                f\'{_soft_timeout_marker("$_rc")}exit "$_rc"\'')
    loop_new = ('                \'| tee -a /logs/agent/opencode-retries.txt >&2; sleep 5; continue; fi; break; done; \'\n'
                '                f\'{_autocontinue_block(_ac_cmd)}\'\n'
                '                f\'{_soft_timeout_marker("$_rc")}exit "$_rc"\'')
    plain_old = ('                f"2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt; "\n'
                 '                f\'{_soft_timeout_marker("${PIPESTATUS[0]}")}\'')
    plain_new = ('                f"2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt; "\n'
                 '                f\'{_autocontinue_plain(_ac_cmd)}\'\n'
                 '                f\'{_soft_timeout_marker("$_rc" if _autocontinue_rounds() > 0 else "${PIPESTATUS[0]}")}\'')
    if s.count(loop_old) == 1:
        s = s.replace(loop_old, loop_new); variant = "relaunch-loop"
    elif s.count(plain_old) == 1:
        s = s.replace(plain_old, plain_new); variant = "plain-pipeline"
    else:
        raise AssertionError("run-command tail not found in either known variant")
    anchor = '        resume_flag = "--continue " if self._resume else ""\n'
    assert s.count(anchor) == 1, "resume_flag anchor not found exactly once"
    s = s.replace(anchor, anchor +
        '        _ac_cmd = (\n'
        '            f"{_soft_timeout_prefix()}opencode --model={self.model_name} run --format=json "\n'
        '            f"--continue {cli_flags_arg}--thinking --dangerously-skip-permissions -- "\n'
        '            + shlex.quote(_autocontinue_msg())\n'
        '        )\n')
    idx = min(i for i in (s.find("\nclass "), s.find("\ndef ")) if i > 0)
    s = s[:idx] + f"\n_AUTOCONTINUE_MSG = {MSG!r}\n" + HELPER + s[idx:]
    open(path, "w").write(s)
    py_compile.compile(path, doraise=True)
    print(f"{path}: patched ({variant})")

patch(os.path.join(sys.argv[1], "opencode.py"))
