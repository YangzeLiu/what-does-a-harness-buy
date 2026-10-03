"""Read-only scan of *-nw* jobs (OpenCode + mini-swe-agent trajectories; CC is covered by noweb_leak_scan_cc.py): tool calls that reference an upstream GitHub URL and whose
observation carries real content (diff/patch/file), i.e. the network block did not hold.

Usage: python3 analysis/noweb_leak_scan_oc_mini.py <dir>   (reads the job trees under <dir>/jobs)"""
import json,glob,re,sys,os
D=sys.argv[1]
URL=re.compile(r"(raw\.githubusercontent\.com/[^\s\"\\\\']+|api\.github\.com/[^\s\"\\\\']+|github\.com/[^\s\"\\\\']*(?:pull|commit|compare|issues)[^\s\"\\\\']*)")
BAD=("Could not resolve","Failed to connect","Transport error","Network is unreachable","Connection refused","Temporary failure","fetch failed","ENOTFOUND","ECONNREFUSED","EAI_AGAIN","No route to host","Connection timed out","timed out","Name or service not known","unable to access","curl: (","URLError","ConnectionError","SSLError","UNEXPECTED_EOF","unexpected eof","Retrying (Retry","returncode\": 35","ECONNRESET")
def success(out):
    if not out: return False
    if any(b in out for b in BAD): return False
    return ("diff --git" in out) or ("Subject: [PATCH]" in out) or (len(out)>1500)
def scan_oc(f):
    hits=[]
    for line in open(f,errors="ignore"):
        if "\"type\":\"tool_use\"" not in line or "github" not in line: continue
        try: d=json.loads(line)
        except Exception: continue
        st=d["part"].get("state",{}); inp=json.dumps(st.get("input",{})); m=URL.search(inp)
        if m and st.get("status")=="completed" and success(str(st.get("output",""))): hits.append((d["part"].get("tool"),m.group(0)[:90],len(str(st.get("output","")))))
    return hits
def scan_cc(f):
    hits=[]; d=json.load(open(f)); steps=d.get("steps",[])
    for s in steps:
        for tc in s.get("tool_calls") or []:
            inp=json.dumps(tc.get("arguments",{})); m=URL.search(inp)
            if not m: continue
            obs=json.dumps(s.get("observation",{}))
            if success(obs): hits.append((tc.get("function_name"),m.group(0)[:90],len(obs)))
    return hits
def scan_mini(f):
    hits=[]; d=json.load(open(f)); msgs=d.get("messages",[])
    for i,mm in enumerate(msgs):
        if mm.get("role")!="assistant": continue
        inp=json.dumps(mm.get("tool_calls",[]))+str(mm.get("content","")); m=URL.search(inp)
        if not m: continue
        out=str(msgs[i+1].get("content","")) if i+1<len(msgs) else ""
        if success(out): hits.append(("bash",m.group(0)[:90],len(out)))
    return hits
res={}
for job in sorted(os.listdir(D+"/jobs")):
    if "-nw" not in job: continue
    for tr in glob.glob(D+"/jobs/"+job+"/*/agent/"):
        if os.path.exists(tr+"opencode.txt"): h=scan_oc(tr+"opencode.txt")
        elif os.path.exists(tr+"mini-swe-agent.trajectory.json"): h=scan_mini(tr+"mini-swe-agent.trajectory.json")
        else: continue  # Claude Code trajectories: see noweb_leak_scan_cc.py (observations must be matched by call id)
        if h:
            rj=os.path.dirname(tr.rstrip("/"))+"/result.json"
            r=json.load(open(rj)) if os.path.exists(rj) else {}
            exc=(r.get("exception_info") or {}).get("exception_type")
            rw=((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            res.setdefault(job,[]).append((os.path.basename(os.path.dirname(tr.rstrip("/"))),exc,rw,h[:3]))
for j,v in res.items():
    print(j,len(v))
    for x in v: print("   ",x)
print("scan done")
