"""Baseline extractors on the raw HTML (CPU): trafilatura and jusText for the extraction comparison (Figure 8). usage: extract_baselines.py <html_table.jsonl> <out.jsonl> [procs]
trafilatura 2.0.0, default settings, plain text:
    trafilatura.extract(html)   (output_format="txt", include_comments=True, include_tables=True, favor_precision=False,
    favor_recall=False, include_formatting=False, include_links=False, include_images=False, deduplicate=False,
    with_metadata=False, url=None, DEFAULT_CONFIG)
jusText 3.0.2, English stoplist, default thresholds (length_low=70, length_high=200, stopwords_low=0.30,
    stopwords_high=0.32, max_link_density=0.2, max_heading_distance=200, no_headings=False), html passed as str:
    "\n".join(p.text for p in justext.justext(html, justext.get_stoplist("English")) if not p.is_boilerplate)
None output (trafilatura returns None when it finds nothing) -> "". Exceptions / 120 s timeouts -> "" + err field.
"""
import json, os, sys, time, signal, platform
from multiprocessing import Pool
TIMEOUT = int(os.environ.get("EXTRACT_TIMEOUT", "120"))
class TO(Exception): pass
def _init():
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))
def _run(fn, html):
    t0 = time.time(); signal.alarm(TIMEOUT)
    try:
        return fn(html) or "", None, time.time() - t0
    except TO:
        return "", "timeout", time.time() - t0
    except Exception as e:
        return "", "%s: %s" % (type(e).__name__, str(e)[:200]), time.time() - t0
    finally:
        signal.alarm(0)
def traf(html):
    import trafilatura
    return trafilatura.extract(html)
def jt(html):
    import justext
    return "\n".join(p.text for p in justext.justext(html, justext.get_stoplist("English")) if not p.is_boilerplate)
def work(r):
    a = _run(traf, r["html"]); b = _run(jt, r["html"])
    return {"gid": r["gid"], "trafilatura": a[0], "justext": b[0], "err_trafilatura": a[1], "err_justext": b[1],
            "sec_trafilatura": round(a[2], 3), "sec_justext": round(b[2], 3)}
def main():
    import trafilatura, justext, lxml
    inp, out = sys.argv[1], sys.argv[2]; procs = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    rows = [json.loads(l) for l in open(inp, encoding="utf-8")]
    t0 = time.time()
    with Pool(procs, initializer=_init, maxtasksperchild=200) as pool:
        res = pool.map(work, rows, chunksize=4)
    res.sort(key=lambda r: r["gid"])
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    meta = {"pages": len(res), "procs": procs, "wall_sec": round(time.time() - t0, 1), "host": platform.node(),
            "trafilatura": trafilatura.__version__, "justext": "3.0.2", "lxml": lxml.__version__,
            "cpu_sec_trafilatura": round(sum(r["sec_trafilatura"] for r in res), 1),
            "cpu_sec_justext": round(sum(r["sec_justext"] for r in res), 1),
            "errors_trafilatura": sum(r["err_trafilatura"] is not None for r in res),
            "errors_justext": sum(r["err_justext"] is not None for r in res),
            "empty_trafilatura": sum(not r["trafilatura"].strip() for r in res),
            "empty_justext": sum(not r["justext"].strip() for r in res)}
    json.dump(meta, open(out + ".meta.json", "w"), indent=1)
    print("EXTRACT_BASELINES_DONE", json.dumps(meta), flush=True)
if __name__ == "__main__":
    main()
