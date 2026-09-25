"""UltraX stage 1: resiliparse text shards -> UltraX inference input.

Input : $UX_SRC/<stem>.jsonl.gz, field "text". For the UltraX row of the main table (10.78B) this is the raw
        resiliparse extraction of the whole source pool (DCLM `resiliparse_extract.yaml` = only the
        resiliparse_extraction_modifier; no length, language or RefinedWeb rule filter, no dedup, no fastText;
        10,319 shards). For the earlier UltraX run (5.99B) it was the RefinedWeb-rule corpus after BFF dedup
        (the input the 7.04B RefinedWeb-rule row was tokenized from).
Output: $UX_WORK/in/<stem>.parquet, single column `original` (what UltraX's inference.py reads; it writes one output
        parquet per input file, so the shard identity survives the round trip). A row is kept iff text.strip() is
        non-empty (the text itself is NOT stripped). Resumable: shards whose output exists are skipped; writes are
        tmp + os.replace.
usage: ux_pool_prep.py <task_id> <n_tasks>   env: UX_SRC, UX_WORK (or WORK_DIR)
"""
import gzip, json, os, sys
import pyarrow as pa
import pyarrow.parquet as pq

R = os.environ.get("WORK_DIR", "")
W = os.environ.get("UX_WORK", os.path.join(R, "dclm_pipeline", "resiliparse_raw_ultrax"))
# default = the resiliparse extraction written by baselines/rule_based/run_resiliparse_pipeline.sh (phase 2)
SRC = os.environ.get("UX_SRC", os.path.join(R, "dclm_pipeline", "resiliparse", "resiliparse_extract",
                                            "resiliparse_extract", "processed_data"))
IN = W + "/in"
tid, nt = int(sys.argv[1]), int(sys.argv[2])
os.makedirs(IN, exist_ok=True)
shards = sorted(f for f in os.listdir(SRC) if f.endswith(".jsonl.gz"))[tid::nt]
done = empty = 0
for k, f in enumerate(shards):
    stem = f[: -len(".jsonl.gz")]
    dst = os.path.join(IN, stem + ".parquet")
    if os.path.exists(dst):
        continue
    texts = []
    with gzip.open(os.path.join(SRC, f), "rt", encoding="utf-8") as g:
        for line in g:
            try:
                t = json.loads(line).get("text") or ""
            except ValueError:
                t = ""
            if t.strip():
                texts.append(t)
            else:
                empty += 1
    tmp = dst + ".tmp%d" % os.getpid()
    pq.write_table(pa.table({"original": texts}), tmp)
    os.replace(tmp, dst)
    done += 1
    if done % 100 == 1:
        print("task %d %d/%d rows_last %d" % (tid, done, len(shards), len(texts)), flush=True)
print("task %d DONE shards %d empty_rows %d" % (tid, done, empty), flush=True)
print("ULTRAX_POOL_PREP_OK")
