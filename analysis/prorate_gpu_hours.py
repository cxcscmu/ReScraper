"""Attribute the GPU hours of a sharded array job to a subset of its shards (Table 7, tab:teacher-cost, Dripper row).

The Dripper teacher ran over the whole source pool, but the SFT set is drawn from a subset of the pool shards, so
its table row is the share of the full run spent on those shards. The main estimate pro-rates the all-attempts
total by each shard's measured Dripper minutes ("global pro-rata ... by minutes"); the other estimates are
cross-checks (by shard count, by pages, per attempt) and a floor (the last successful attempt's own minutes).
The RePro row is pro-rated the same way by page count, outside this script: the RePro job's total times the
share of its rewrite candidates that lie in the SFT shards.

Inputs (one directory):
  shard_stats.json     {stem: {"sid": global shard index, "pairs": pages, "empty": empty outputs, "mins": minutes}},
                       parsed from Dripper's per-shard log line "[Shard N] Done. X pairs ... (K with empty output, T min)"
  subset_stems.txt     the shards the SFT set is drawn from, one stem per line
  done_mtimes_utc.txt  "<ISO time> <stem>.done" per shard (modification time of each shard's completion marker)
  sacct_D.txt          `sacct -D -X -n -P --format=JobID,JobIDRaw,JobName,State,Start,End,ElapsedRaw,AllocTRES`
                       for the array jobs of the run (further columns are ignored)
Shard i belongs to task i // shards_per_task of an array (tasks cover the sorted shard list in fixed ranges).
usage: python3 prorate_gpu_hours.py <input dir> --shards-per-task ARRAY_JOBID=N[,ARRAY_JOBID=N...]
"""
import json, re, collections, datetime as dt, sys
if len(sys.argv) != 4 or sys.argv[2] != "--shards-per-task":
    sys.exit(__doc__)
SP=sys.argv[1]
res=json.load(open(SP+"/shard_stats.json"))
S=set(l.strip() for l in open(SP+"/subset_stems.txt") if l.strip())
idx2stem={r["sid"]:s for s,r in res.items()}
assert len(idx2stem)==len(res)
N=len(res)
mt={}
for l in open(SP+"/done_mtimes_utc.txt"):
    t,f=l.split(); mt[f[:-5]]=dt.datetime.fromisoformat(t)
SPT={k:int(v) for k,v in (x.split("=") for x in sys.argv[3].split(","))}
att=[]
for l in open(SP+"/sacct_D.txt"):
    f=l.rstrip("\n").split("|")
    jid=f[0]
    if "[" in jid: continue
    arr,task=jid.split("_"); task=int(task)
    m=re.search(r"gres/gpu=(\d+)",f[7]); g=int(m.group(1)) if m else 0
    if f[4]=="None": continue
    st=dt.datetime.fromisoformat(f[4]); en=dt.datetime.fromisoformat(f[5])
    lo=task*SPT[arr]; hi=min(lo+SPT[arr],N)
    att.append(dict(arr=arr,task=task,st=st,en=en,el=int(f[6]),g=g,lo=lo,hi=hi,state=f[3].split()[0],done=[]))
tot=sum(a["el"]*a["g"] for a in att)/3600
print("attempts",len(att),"GPU-h %.1f"%tot)
# assign each shard to attempt(s) whose range contains idx and window contains marker mtime
unassigned=0; amb=0
for i,s in idx2stem.items():
    t=mt[s]; c=[a for a in att if a["lo"]<=i<a["hi"] and a["st"]<=t<=a["en"]+dt.timedelta(seconds=10)]
    if not c: unassigned+=1; continue
    if len(c)>1: amb+=1
    for a in c: a["done"].append((s,1.0/len(c)))
print("shards unassigned",unassigned,"ambiguous",amb)
# method A: attempt GPU-h split over its completed shards by per-shard minutes; wasted attempts by range minutes share
attA=0.0; wasted=0.0; wasted_sft=0.0
for a in att:
    h=a["el"]*a["g"]/3600
    if a["done"]:
        allm=sum(res[s]["mins"]*w for s,w in a["done"]); sftm=sum(res[s]["mins"]*w for s,w in a["done"] if s in S)
        attA+=h*sftm/allm if allm>0 else 0
    else:
        rng=[idx2stem[i] for i in range(a["lo"],a["hi"])]
        allm=sum(res[s]["mins"] for s in rng); sftm=sum(res[s]["mins"] for s in rng if s in S)
        wasted+=h; wasted_sft+=h*sftm/allm
        attA+=h*sftm/allm
print("method A (attempt->its completed shards, wasted attempts by range): SFT GPU-h %.1f ; attempts with no completed shard: %.1f GPU-h (of which SFT-attributed %.1f)"%(attA,wasted,wasted_sft))
# method B: every attempt split over its whole task range by per-shard minutes
attB=0.0
for a in att:
    h=a["el"]*a["g"]/3600
    rng=[idx2stem[i] for i in range(a["lo"],a["hi"])]
    allm=sum(res[s]["mins"] for s in rng); sftm=sum(res[s]["mins"] for s in rng if s in S)
    attB+=h*sftm/allm
print("method B (attempt->whole task range, by minutes): SFT GPU-h %.1f"%attB)
# global pro-rata
tp=sum(r["pairs"] for r in res.values()); tm=sum(r["mins"] for r in res.values())
sp=sum(res[s]["pairs"] for s in S); sm=sum(res[s]["mins"] for s in S)
print("global pro-rata: by shards %.1f  by pages %.1f  by minutes %.1f ; direct last-attempt minutes %.1f"%(tot*len(S)/N, tot*sp/tp, tot*sm/tm, sm/60))
# which arrays completed SFT shards
by=collections.Counter(); byall=collections.Counter()
for a in att:
    for s,w in a["done"]:
        byall[a["arr"]]+=w
        if s in S: by[a["arr"]]+=w
print("shards completed per array (all / SFT):", {k:(round(byall[k]),round(by[k])) for k in byall})
