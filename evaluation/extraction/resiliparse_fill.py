"""Fallback resiliparse text for page-table rows whose pool `resiliparse` is null (DCLM env, resiliparse 0.16.0), with
the settings of the pool extraction (DCLM resiliparse_extraction_modifier, as used by the pool baselines):
extract_plain_text(HTMLTree.parse(html), main_content=True, alt_texts=False, preserve_formatting=True), 60 s timeout.
Computed for every page (on the first 960 pages it reproduces the pool text byte for byte); ext_metrics uses it only
where the pool text is null. usage: resiliparse_fill.py <html_table> <out.jsonl> [procs]"""
import sys, json, os, signal
from multiprocessing import Pool
class TO(Exception): pass
def _init(): signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))
def work(r):
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.html import HTMLTree
    signal.alarm(60)
    try: t, e = extract_plain_text(HTMLTree.parse(r["html"]), main_content=True, alt_texts=False, preserve_formatting=True), None
    except Exception as ex: t, e = None, type(ex).__name__
    finally: signal.alarm(0)
    return {"gid": r["gid"], "resiliparse_refill": t, "err": e}
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8, initializer=_init) as p: res = p.map(work, rows, chunksize=8)
tmp = sys.argv[2] + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    for r in sorted(res, key=lambda r: r["gid"]): f.write(json.dumps(r, ensure_ascii=False) + "\n")
os.replace(tmp, sys.argv[2]); print("RESILIPARSE_FILL_DONE", len(res), "errors", sum(r["err"] is not None for r in res))
