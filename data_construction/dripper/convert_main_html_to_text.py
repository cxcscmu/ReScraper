#!/usr/bin/env python3
"""Step 2 of the Dripper teacher run: convert Dripper's main_html into DCLM-ready text shards.

Input : step1_inference/<shard>.jsonl   with {"input": raw_html, "main_html": ...}
Output: step2_text/<shard>.jsonl.gz     with {"text": <plain text>}   (content key "text", and DCLM's
        process_no_ray.py globs *.jsonl.gz)

Rows keep the step-1 order; pages whose main_html is empty, whose conversion fails or times out, or whose text is
empty are dropped (lib/pool_join.py re-aligns step-2 rows with step-1 rows).

Text conversion is `webkit_txt` from lib/render.py, the same renderer the student reads, so the Dripper text carries
the same text convention as every other ReScraper input and target.
usage: convert_main_html_to_text.py --in_dir <step1> --out_dir <step2> [--task_id i --num_tasks n] [--workers k]
env: DOC_TIMEOUT (per-page conversion timeout in seconds, default 60)
"""
import argparse, gzip, json, os, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import render


def _init():
    render._load_converter()


def webkit_txt(html: str) -> str:
    return render.webkit_txt(html)

class _Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise _Timeout()

# A few pathological docs make webkit_txt hang (regex backtracking); without a per-doc
# guard one such doc stalls its whole shard forever. 60s is ~1000x the normal per-doc cost.
DOC_TIMEOUT = int(os.environ.get("DOC_TIMEOUT", "60"))

def one(main_html):
    if not main_html or not main_html.strip():
        return None
    import signal
    old_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(DOC_TIMEOUT)
    try:
        t = webkit_txt(main_html)
    except _Timeout:
        return None          # skip the offending doc, keep the shard moving
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return t if t and t.strip() else None

def convert_shard(pool, src, dst):
    mains = []
    with open(src, "rt", errors="ignore") as f:
        for line in f:
            try: mains.append(json.loads(line).get("main_html", ""))
            except Exception: mains.append("")
    kept = 0
    tmp = dst + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for t in pool.imap(one, mains, chunksize=16):
            if t is None:
                continue
            out.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            kept += 1
    os.replace(tmp, dst)
    return len(mains), kept

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--num_tasks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    shards = sorted(f for f in os.listdir(a.in_dir) if f.endswith(".jsonl"))
    mine = shards[a.task_id::a.num_tasks]
    # skip up front so a restarted task does not re-stat every shard inside the loop
    todo = [s for s in mine
            if not os.path.exists(os.path.join(a.out_dir, s[:-len(".jsonl")] + ".jsonl.gz"))]
    print(f"task {a.task_id}/{a.num_tasks}: {len(todo)}/{len(mine)} shards to do", flush=True)
    import time
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:      # one pool for ALL shards
        for i, s in enumerate(todo):
            base = s[:-len(".jsonl")]
            dst = os.path.join(a.out_dir, base + ".jsonl.gz")
            if os.path.exists(dst):
                continue
            n, k = convert_shard(pool, os.path.join(a.in_dir, s), dst)
            el = time.time() - t0
            print(f"[{i+1}/{len(todo)}] {base}: {n} -> {k} "
                  f"({100*k/max(1,n):.1f}% kept) {el/(i+1):.1f}s/shard", flush=True)
    print("CONVERT_TASK_DONE", flush=True)
