"""Write the raw resiliparse text (UltraX parquet column `original`, row order kept, empty rows kept as "") of the 376
stems to $DIV_DIR/raw/<stem>.jsonl.gz, for the FineWeb-rule env (datatrove 0.2.0), which has no pyarrow.
usage: extract_raw.py"""
import gzip, json, os, sys, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("WORK_DIR") or sys.exit("set WORK_DIR")
DIV = os.environ.get("DIV_DIR") or os.path.join(os.environ.get("EVAL_DIR") or os.path.join(WORK, "eval"), "diversity")
UX = os.environ.get("ULTRAX_RERUN_DIR") or os.path.join(WORK, "dclm_pipeline", "resiliparse_raw_ultrax")
os.makedirs(DIV + "/raw", exist_ok=True)
for s in [l.strip() for l in open(HERE + "/stems376.txt") if l.strip()]:
    o = "%s/raw/%s.jsonl.gz" % (DIV, s)
    if os.path.exists(o): continue
    t = pq.read_table("%s/post/%s_processed.parquet" % (UX, s), columns=["original"]).column("original").to_pylist()
    with gzip.open(o + ".tmp", "wt", encoding="utf-8") as f:
        for x in t: f.write(json.dumps({"text": x or ""}) + "\n")
    os.replace(o + ".tmp", o)
print("EXTRACT_RAW_DONE")
