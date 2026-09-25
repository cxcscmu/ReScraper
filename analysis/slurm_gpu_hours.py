"""GPU hours of SLURM jobs from accounting, all attempts included (Table 7, tab:teacher-cost).

GPU-h = sum over every attempt of ElapsedRaw x gres/gpu / 3600, from `sacct -D -X` (-D keeps every attempt of a
requeued job, since interrupted attempts did real work; -X keeps one allocation record per attempt). The
completed-only total is printed for reference. The GPU model is not in the accounting record; check the nodes
with `sinfo` (the paper's jobs all ran on H200 nodes).

Jobs are selected by any combination of
  --name NAME             exact job name (repeatable), e.g. the name of each teacher / SFT / inference job
  --name-prefix P         job-name prefix (repeatable), e.g. the per-corpus pretraining jobs of one scale
  --submit-contains S     substring of the SubmitLine (repeatable), e.g. "INFER_OUT_DIR=<corpus text dir>" to
                          catch the main array and every backfill job that wrote one corpus
  --job ID                job id (repeatable)
within the window --start/--end (sacct -S/-E) for --user (default: the current user).
For a job that ran over more pages than a table row counts, pro-rate its total with prorate_gpu_hours.py.
usage: slurm_gpu_hours.py --start YYYY-MM-DD [--end YYYY-MM-DD] [--user U] [selectors ...]
"""
import argparse, collections, getpass, re, subprocess

F = "JobIDRaw,JobID,JobName%60,State,ElapsedRaw,AllocTRES%90,Start,End,SubmitLine%1500"


def recs(args):
    out = subprocess.run(["sacct", "-D", "-X", "-n", "-P", "--format=" + F] + args,
                         capture_output=True, text=True).stdout.strip().splitlines()
    rows = []
    for l in out:
        p = l.split("|", 8)
        if len(p) < 9: continue
        m = re.search(r"gres/gpu=(\d+)", p[5]); g = int(m.group(1)) if m else 0
        rows.append(dict(raw=p[0], jid=p[1], name=p[2], st=p[3].split()[0], el=int(p[4] or 0), g=g,
                         start=p[6], end=p[7], sub=p[8]))
    return rows


def gh(rows): return sum(r["el"] * r["g"] for r in rows) / 3600


def summ(label, rows):
    if not rows:
        print(f"== {label}: no records"); return
    by = collections.defaultdict(float); cnt = collections.Counter()
    for r in rows: by[r["st"]] += r["el"] * r["g"] / 3600; cnt[r["st"]] += 1
    print(f"== {label}: records {len(rows)} (unique raw ids {len(set(r['raw'] for r in rows))}), "
          f"GPU-h all attempts {gh(rows):.1f} | completed-only {gh([r for r in rows if r['st'] == 'COMPLETED']):.1f}")
    print("   by state:", {k: (cnt[k], round(v, 1)) for k, v in by.items()})
    print("   gpus/job:", dict(collections.Counter(r["g"] for r in rows)), "| elapsed h min/max:",
          round(min(r["el"] for r in rows) / 3600, 2), round(max(r["el"] for r in rows) / 3600, 2))
    print("   first start:", min(r["start"] for r in rows), "| last end:", max(r["end"] for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True); ap.add_argument("--end", default=None)
    ap.add_argument("--user", default=getpass.getuser())
    ap.add_argument("--name", action="append", default=[]); ap.add_argument("--name-prefix", action="append", default=[])
    ap.add_argument("--submit-contains", action="append", default=[]); ap.add_argument("--job", action="append", default=[])
    a = ap.parse_args()
    window = ["-u", a.user, "-S", a.start] + (["-E", a.end] if a.end else [])
    allj = recs(window) if (a.name_prefix or a.submit_contains) else []
    for n in a.name:
        summ("job name " + n, recs(window + ["--name=" + n]))
    for p in a.name_prefix:
        rs = [r for r in allj if r["name"].startswith(p)]
        summ("job names starting with " + p, rs)
        byn = collections.defaultdict(list)
        for r in rs: byn[r["name"]].append(r)
        for n, x in sorted(byn.items()):
            print(f"   {n:40s} all {gh(x):7.1f} completed {gh([r for r in x if r['st'] == 'COMPLETED']):7.1f} "
                  f"{dict(collections.Counter(r['st'] for r in x))}")
    for s in a.submit_contains:
        rs = [r for r in allj if s in r["sub"]]
        summ("SubmitLine contains " + s, rs)
        byn = collections.defaultdict(list)
        for r in rs: byn[r["name"]].append(r)
        for n, x in byn.items():
            print(f"   {n:32s} runs {len(x):3d}  GPU-h {gh(x):7.1f}  completed-only "
                  f"{gh([r for r in x if r['st'] == 'COMPLETED']):7.1f}  states {dict(collections.Counter(r['st'] for r in x))}")
    for j in a.job:
        summ("job " + j, recs(["-j", j]))


if __name__ == "__main__":
    main()
