"""UltraX stage 4: post-processed parquet -> corpus text shards.

One output .jsonl.gz per input parquet, same stem (a trailing `_processed` is dropped). Only `cleaned` is kept.
Pages UltraX emptied (remove_all, or a clean that left nothing) are dropped and counted: that deletion IS the
cleaning step being measured. No length filter anywhere (MINCH = 1 drops only empties).
For the 10.78B UltraX corpus the input is the unfiltered resiliparse extraction, so BFF dedup follows
(corpus/dedup_bff.sh), exactly as for our corpus. For the earlier 5.99B run the input had already been
rule-filtered and deduplicated, so its text/ shards were tokenized directly.
Resumable: shards whose output exists are skipped; writes are tmp + os.replace. After an abnormal termination, test
every output with `gzip -t` before deduplicating (a crash can leave a complete-looking file with lost data blocks).
usage: ux_build_leg.py <task_id> <n_tasks>   env: UX_WORK (or WORK_DIR)
"""
import gzip, json, os, sys
import pyarrow.parquet as pq

R = os.environ.get("WORK_DIR", "")
W = os.environ.get("UX_WORK", os.path.join(R, "dclm_pipeline", "resiliparse_raw_ultrax"))
SRC, DST = W + "/post", W + "/text"
MINCH = 1  # only empties are dropped; this pipeline has no length filter anywhere
tid, nt = int(sys.argv[1]), int(sys.argv[2])
os.makedirs(DST, exist_ok=True)
files = sorted(f for f in os.listdir(SRC) if f.endswith(".parquet"))[tid::nt]
kept = dropped = shards = 0
for f in files:
    stem = f[: -len(".parquet")]
    if stem.endswith("_processed"):
        stem = stem[: -len("_processed")]
    out = os.path.join(DST, stem + ".jsonl.gz")
    if os.path.exists(out):
        continue
    t = pq.read_table(os.path.join(SRC, f), columns=["cleaned"])
    tmp = out + ".tmp%d" % os.getpid()
    k = d = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as g:
        for v in t.column("cleaned"):
            s = (v.as_py() or "").strip()
            if len(s) < MINCH:
                d += 1
                continue
            g.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
            k += 1
    os.replace(tmp, out)
    kept += k; dropped += d; shards += 1
    if shards % 200 == 1:
        print("task %d: %d shards, kept %d, emptied %d" % (tid, shards, kept, dropped), flush=True)
print("task %d DONE shards %d kept %d emptied %d" % (tid, shards, kept, dropped), flush=True)
