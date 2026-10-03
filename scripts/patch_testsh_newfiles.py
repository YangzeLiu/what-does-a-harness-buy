#!/usr/bin/env python3
"""Make test.sh clear files that the gold test patch *adds* before applying it.

Harbor's SWE-bench task template resets the test files the gold test patch
touches before applying it, but its cleanup loop only lists modified files
(`--- a/<path>`); on some tasks the files the test patch *adds* (`new file
mode`) are missing. If the agent created a file at that path, `git apply`
refuses ("already exists in working directory") and the trial has no grade
(django-13837: 21 such trials by 2026-09-08).

Fix: after each existing cleanup loop, insert a loop that deletes every
new-file path unconditionally (such a path cannot exist at the base commit,
so deleting it is always correct) and drops it from the index in case the
agent staged it. Idempotent (marker comment); backup test.sh.orig-newfile.

Usage: patch_testsh_newfiles.py <task_dir>...
"""
import re, shutil, sys, os

MARK = "# newfile-cleanup: paths the gold test patch adds must not pre-exist"

def patch(task_dir):
    t = os.path.join(task_dir, "tests", "test.sh")
    if not os.path.exists(t):
        return "no test.sh"
    s = open(t).read()
    if MARK in s:
        return "already patched"
    new = re.findall(r"diff --git a/(\S+) b/\S+\nnew file mode", s)
    if not new:
        return "no new files"
    loops = list(re.finditer(r"for path in [^;\n]+; do\n(?:.*\n)*?done\n", s))
    if not loops:
        return "NO LOOP"
    block = (MARK + "\n" + "for path in " + " ".join(new) + "; do\n"
             "    git rm -q --cached -- \"$path\" >/dev/null 2>&1 || true\n"
             "    rm -rf -- \"$path\"\n"
             "done\n")
    out = s
    for m in reversed(loops):
        out = out[:m.end()] + block + out[m.end():]
    bak = t + ".orig-newfile"
    if not os.path.exists(bak):
        shutil.copy2(t, bak)
    open(t, "w").write(out)
    return "patched (%d loops, +%s)" % (len(loops), ",".join(new))

if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(os.path.basename(d.rstrip("/")), patch(d))
