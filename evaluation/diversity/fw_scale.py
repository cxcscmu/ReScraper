"""FineWeb-rule on the raw resiliparse text of the 376 shards ($DIV_DIR/raw/<stem>.jsonl.gz from extract_raw.py = the
UltraX parquet column `original`, the raw resiliparse extraction every baseline reads). Chain = heldout_pipeline
stages/s2_fw_rule.py (lib/fw_rule_chain.py, datatrove 0.2.0): blank-line strip -> LanguageFilter(en 0.65) ->
GopherRepetition -> GopherQuality -> C4(no terminal punct) -> FineWebQuality.
DEVIATION: no URLFilter (the parquet has no URL; on the 5,000 held-out pages it removed 27 pages).
Output $DIV_DIR/fw/<stem>.jsonl.gz ({"text"} of kept docs, file order).  usage: fw_scale.py <nproc>"""
import gzip, json, os, sys
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "heldout_pipeline", "lib"))
import fw_rule_chain as G
WORK = os.environ.get("WORK_DIR") or sys.exit("set WORK_DIR")
D = os.environ.get("DIV_DIR") or os.path.join(os.environ.get("EVAL_DIR") or os.path.join(WORK, "eval"), "diversity")
FILTERS = None
def one(stem):
    global FILTERS
    out = "%s/fw/%s.jsonl.gz" % (D, stem)
    if os.path.exists(out): return stem, -1, -1
    if FILTERS is None:
        _, FILTERS = G.build()
    with gzip.open("%s/raw/%s.jsonl.gz" % (D, stem), "rt", encoding="utf-8") as f:
        texts = [json.loads(l)["text"] for l in f]
    res = G.res; Document = G.Document; kept = []
    for i, t in enumerate(texts):
        if not (t and t.strip()): continue
        t = "\n".join(l for l in t.splitlines() if l.strip())
        doc = Document(text=t, id=str(i), metadata={})
        ok = True
        for name, f in FILTERS:
            try:
                ok, why = res(f.filter(doc))
            except Exception:
                ok = False
            if not ok: break
        if ok and doc.text.strip():
            kept.append(doc.text)
    with gzip.open(out + ".tmp", "wt", encoding="utf-8") as f:
        for t in kept: f.write(json.dumps({"text": t}) + "\n")
    os.replace(out + ".tmp", out)
    return stem, len(texts), len(kept)
if __name__ == "__main__":
    os.makedirs(D + "/fw", exist_ok=True)
    stems = [l.strip() for l in open(HERE + "/stems376.txt") if l.strip()]
    tin = tk = 0
    with Pool(int(sys.argv[1])) as p:
        for j, (s, a, b) in enumerate(p.imap_unordered(one, stems)):
            tin += max(a, 0); tk += max(b, 0)
            if j % 25 == 0: print(j, s, a, b, "cum in %d kept %d" % (tin, tk), flush=True)
    print("FW_SCALE_DONE docs_in %d kept %d" % (tin, tk), flush=True)
