"""CC no-web leak scan: match each WebFetch/Bash tool call that names a GitHub URL to its
observation by source_call_id, and report calls whose observation looks like real fetched content.

Usage: python3 analysis/noweb_leak_scan_cc.py <dir>   (reads <dir>/jobs/cc-*-nw*/*/agent/trajectory.json)"""
import json,glob,re,sys,os
D=sys.argv[1]
URL=re.compile(r"(raw\.githubusercontent\.com/[^\s\"\\\\']+|api\.github\.com/[^\s\"\\\\']+|github\.com/[^\s\"\\\\']*(?:pull|commit|compare|issues)[^\s\"\\\\']*)")
BAD=("Could not resolve","Failed to connect","Transport error","Network is unreachable","Connection refused","Temporary failure","fetch failed","ENOTFOUND","ECONNREFUSED","EAI_AGAIN","No route to host","Connection timed out","timed out","Name or service not known","unable to access","curl: (","URLError","ConnectionError","SSLError","UNEXPECTED_EOF","unexpected eof","Retrying (Retry","Unable to fetch","Domain","not allowed","blocked","proxy","ECONNRESET","socket hang up","TLS","certificate")
def looks_real(c):
    if not c or any(b.lower() in c.lower() for b in BAD): return False
    return ("diff --git" in c) or ("Subject: [PATCH]" in c) or len(c)>1500
tot={}; hits={}; status_samples={}
for job in sorted(os.listdir(D+"/jobs")):
    if "-nw" not in job or not job.startswith("cc-"): continue
    for f in glob.glob(D+"/jobs/"+job+"/*/agent/trajectory.json"):
        try: d=json.load(open(f))
        except Exception: continue
        steps=d.get("steps",[])
        obs={}
        for s in steps:
            for r in (s.get("observation") or {}).get("results") or []:
                obs[r.get("source_call_id")]=str(r.get("content",""))
        for s in steps:
            for tc in s.get("tool_calls") or []:
                if tc.get("function_name") not in ("WebFetch","Bash"): continue
                inp=json.dumps(tc.get("arguments",{})); m=URL.search(inp)
                if not m: continue
                c=obs.get(tc.get("tool_call_id"),"")
                tot[job]=tot.get(job,0)+1
                if looks_real(c):
                    hits.setdefault(job,[]).append((f.split("/")[-3],tc["function_name"],m.group(0)[:80],len(c),c[:160].replace("\n"," ")))
                else:
                    status_samples.setdefault(job,c[:120].replace("\n"," "))
for j in sorted(tot):
    print(j,"github-url calls:",tot[j],"look-real:",len(hits.get(j,[])),"| typical blocked reply:",status_samples.get(j,"")[:110])
    for h in hits.get(j,[]): print("    HIT",h)
print("scan done")
