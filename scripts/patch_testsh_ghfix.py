#!/usr/bin/env python3
"""Make the SWE-bench verifier parser grade offline.

The parser.py heredoc inside every task's tests/test.sh calls swebench's make_test_spec(),
which fetches the repo's requirements.txt / environment.yml from raw.githubusercontent.com
(swebench.harness.test_spec.python.get_requirements / get_environment_yml) even though the
grading path only needs FAIL_TO_PASS / PASS_TO_PASS / repo / instance_id.  GitHub is flaky
from host-b (2026-09-11: pylint-4604, django-11276, django-11451 verifier_incomplete within
one hour), so stub the two fetchers before make_test_spec() runs.  Verified offline locally
(socket blocked): make_test_spec(django-11451) still yields f2p 6 / p2p 45.

Usage: patch_testsh_ghfix.py <task_root> [<task_root> ...]   (idempotent; backup test.sh.orig-ghfix-20260911)
"""
import sys, pathlib, shutil

MARKER = "# agent-harness 2026-09-11 ghfix"
ANCHOR = "from swebench.harness.test_spec.test_spec import make_test_spec\n"
# NOTE: the heredoc is unquoted (cat > parser.py <<EOF): no '$' or backticks below.
STUB = (
    ANCHOR
    + MARKER + ": grading only needs FAIL_TO_PASS/PASS_TO_PASS; stub the\n"
    "# requirements/environment.yml fetch from raw.githubusercontent.com so verification is offline.\n"
    "import swebench.harness.test_spec.python as _swb_py\n"
    "_swb_py.get_requirements = lambda *a, **k: \"\"\n"
    "_swb_py.get_environment_yml = lambda *a, **k: \"\"\n"
)

def patch(path: pathlib.Path) -> str:
    s = path.read_text()
    if MARKER in s:
        return "skip"
    if s.count(ANCHOR) != 1:
        return "no-anchor"
    bak = path.with_name("test.sh.orig-ghfix-20260911")
    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_text(s.replace(ANCHOR, STUB))
    return "patched"

def main():
    counts = {}
    for root in sys.argv[1:]:
        for p in sorted(pathlib.Path(root).glob("*/tests/test.sh")):
            r = patch(p)
            counts[r] = counts.get(r, 0) + 1
            if r == "no-anchor":
                print("no-anchor:", p)
    print(counts)

if __name__ == "__main__":
    main()
