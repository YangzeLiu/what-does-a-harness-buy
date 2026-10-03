#!/usr/bin/env python3
"""Add the no-web agent-phase network policy to every Terminal-Bench 2.1 task.toml (idempotent).

Usage: python3 tb21_patch_noweb.py <tasks_root> [--hosts api.deepseek.com,172.17.0.1]
Inserts, right after the `[agent]` header of each <task>/task.toml:
    network_mode = "allowlist"
    allowed_hosts = [...]
Only the agent phase is restricted (Harbor 0.22 applies [agent] policy to agent.run() only);
[verifier]/[environment] are untouched so the 82/89 test.sh that install uv+pytest keep internet.
Prints per-task status; never rewrites a file that already carries network_mode.
"""
import os, re, sys
root = sys.argv[1]
hosts = "api.deepseek.com,172.17.0.1"
if "--hosts" in sys.argv:
    hosts = sys.argv[sys.argv.index("--hosts") + 1]
block = 'network_mode = "allowlist"\nallowed_hosts = [%s]\n' % ", ".join('"%s"' % h for h in hosts.split(","))
done = skipped = missing = 0
for name in sorted(os.listdir(root)):
    p = os.path.join(root, name, "task.toml")
    if not os.path.isfile(p):
        continue
    s = open(p).read()
    if "network_mode" in s:
        skipped += 1
        continue
    m = re.search(r"^\[agent\]\n", s, re.M)
    if not m:
        print("NO [agent] SECTION:", name); missing += 1
        continue
    s = s[: m.end()] + block + s[m.end():]
    open(p, "w").write(s)
    done += 1
print(f"patched {done}, already had network_mode {skipped}, no [agent] section {missing}")
