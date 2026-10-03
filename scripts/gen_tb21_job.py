#!/usr/bin/env python3
"""Derive a Harbor job YAML for Terminal-Bench 2.1 (tasks_tb21) from an existing Qwen3.6 job config.

Usage: python scripts/gen_tb21_job.py --arm oc|cc|mini --job NAME --template PATH [--conc 8]
       [--tasks all|t1,t2,...] [--tasks-root tasks_tb21] [--setup-mult 6.0] [--agent-mult 6.0|12.0]
Keeps the template's agent block (model, provider, step caps) verbatim; replaces job_name, jobs_dir,
concurrency, multipliers and the task list. Asserts the 300-step cap is present. Writes <job>.yaml (0600)
into the current directory and prints only job_name / task count / caps -- never the key values.

The paper's Terminal-Bench jobs were derived from the second host's Qwen3.6 configs
oc-qwen36-hard45-nw-rg-k2 (oc), cc-qwen36-pool447-nw-r2 (cc) and mini-qwen36-pool447-nw-r2 (mini),
which are not in jobs/.  The closest shipped configs are jobs/oc-qwen36-pool447-nw-rg.json,
jobs/cc-qwen36-pool447-nw.json and jobs/mini-qwen36-pool447-nw.json (a JSON job config is valid
YAML).  jobs_dir is $HARNESS_JOBS_ROOT, else results/raw/jobs.
"""
import argparse, os, stat, sys, yaml
A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ("oc", "cc", "mini")
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=ARMS); ap.add_argument("--job", required=True)
ap.add_argument("--conc", type=int, default=8); ap.add_argument("--tasks", default="all")
ap.add_argument("--tasks-file", default=None, help="file with one task name per line (overrides --tasks)")
ap.add_argument("--tasks-root", default=os.path.join(A, "tasks_tb21"))
ap.add_argument("--setup-mult", type=float, default=6.0); ap.add_argument("--agent-mult", type=float, default=None)
ap.add_argument("--verifier-mult", type=float, default=3.0, help="TB verifier timeout_sec is 900 for most tasks; x3 covers a loaded host")
ap.add_argument("--template", required=True, help="Qwen3.6 job config of the same harness (YAML or JSON)")
a = ap.parse_args()
tpl = a.template
cfg = yaml.safe_load(open(tpl))
names = sorted(d for d in os.listdir(a.tasks_root) if os.path.isfile(os.path.join(a.tasks_root, d, "task.toml")))
if a.tasks_file:
    a.tasks = ",".join(l.strip() for l in open(a.tasks_file) if l.strip())
if a.tasks != "all":
    want = a.tasks.split(","); missing = [t for t in want if t not in names]
    assert not missing, f"unknown tasks: {missing}"; names = want
cfg["job_name"] = a.job; cfg["jobs_dir"] = os.environ.get("HARNESS_JOBS_ROOT", os.path.join("results", "raw", "jobs")); cfg["n_concurrent_trials"] = a.conc
cfg["n_attempts"] = 1; cfg["agent_setup_timeout_multiplier"] = a.setup_mult; cfg["verifier_timeout_multiplier"] = a.verifier_mult
cfg["agent_timeout_multiplier"] = a.agent_mult if a.agent_mult is not None else (12.0 if a.arm == "cc" else 6.0)
cfg["tasks"] = [{"path": os.path.join(a.tasks_root, n)} for n in names]
# verifier phase: 82/89 test.sh run `curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh` then uvx; the installer
# honours INSTALLER_DOWNLOAD_URL, so point it at the host-b 8090 mirror (GitHub release downloads are flaky from host-b:
# curl 56 in the first smoke), and pull pytest from the tuna PyPI mirror. `uvx -p 3.13` then fetches CPython from
# github.com/astral-sh/python-build-standalone (stalled 25 min in smoke2) -> UV_PYTHON_INSTALL_MIRROR at 8090/pbs holds
# the exact 20251014 tarballs uv 0.9.5 resolves for -p 3.12 / -p 3.13 (`uv python list --show-urls`).
venv = (cfg.get("verifier") or {}).get("env") or {}
venv.update({"INSTALLER_DOWNLOAD_URL": "http://172.17.0.1:8090/uv-0.9.5",
             "UV_PYTHON_INSTALL_MIRROR": "http://172.17.0.1:8090/pbs",
             "UV_DEFAULT_INDEX": "https://pypi.tuna.tsinghua.edu.cn/simple", "UV_HTTP_TIMEOUT": "300"})
cfg["verifier"] = {**(cfg.get("verifier") or {}), "env": venv}
ag = cfg["agents"][0]; kw = ag.get("kwargs", {}) or {}
if a.arm == "oc":
    cap = kw["opencode_config"]["agent"]["build"]["steps"]
elif a.arm == "cc":
    cap = kw["max_turns"]
else:
    cap = kw["config"]["agent"]["step_limit"]
assert cap == 300, f"step cap in template is {cap}, expected 300"
out = f"{a.job}.yaml"
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)
print(f"wrote {out}: job={a.job} arm={a.arm} agent={ag['name']} model={ag.get('model_name')} tasks={len(names)} conc={a.conc} "
      f"setup_mult={cfg['agent_setup_timeout_multiplier']} verifier_mult={cfg['verifier_timeout_multiplier']} agent_mult={cfg['agent_timeout_multiplier']} step_cap={cap}")
