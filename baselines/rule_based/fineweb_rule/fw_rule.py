"""FineWeb-rule text on a 60-shard subsample of the DCLM pool.

Input: raw HTML after ONLY DCLM's language + length filters
  ($FW_SRC, default $HTML_POOL_DIR/step3b_length_only/dclm_baseline_refinedweb_post_lang_length_only).
Shards: sorted file list, random.Random(3).shuffle, first 60.
Extraction: resiliparse extract_plain_text(HTMLTree.parse(html), main_content=True, alt_texts=False,
  preserve_formatting=True), 60 s SIGALRM, empty/failed dropped (identical to baselines/prox_c/proxc_pool.py).
FineWeb rule stack = datatrove 0.2.0 (the FineWeb-release version) filters, in the order of datatrove's
  examples/fineweb.py, with its arguments:
    URLFilter()  -> [extraction]  -> LanguageFilter(en, 0.65) -> GopherRepetitionFilter() -> GopherQualityFilter()
    -> C4QualityFilter(filter_no_terminal_punct=False) (line-level edits applied) -> FineWebQualityFilter()
  (FineWeb extracts with trafilatura; here resiliparse, the extractor shared by every rule baseline.)
  LanguageFilter uses fastText lid.176.bin (the file datatrove downloads; LID_MODEL, default the DCLM copy).
  FORMAT ADAPTER: resiliparse (preserve_formatting=True) separates blocks with blank lines, trafilatura does not.
  Gopher's line rules count every "" line as a duplicate / non-bullet line (dup_line_frac > 0.3 then drops ~2/3 of
  pages for formatting alone), so before the filters blank/whitespace-only lines are removed
  ("\n".join(l for l in lines if l.strip())), i.e. trafilatura's one-block-per-line layout. How many pages the
  repetition filter would have dropped on the raw layout is logged as diag_raw_gopher_rep_drop.
NOT applied: FineWeb's MinHash dedup, PII formatting (dedup deliberately skipped).
Sample: per shard (in shuffled order) survivors in shard order, random.Random(4).shuffle, first 100; cap 5000.
The module-level objects (build, res, extract, Document, LID) are also used page by page by the held-out pipeline,
which executes this file up to the `if __name__ == "__main__":` line.
usage: fw_rule.py <out_dir> <procs>
"""
import gzip, json, os, random, signal, sys, collections
from multiprocessing import Pool

R = os.environ.get("WORK_DIR", "")
POOL = os.environ.get("HTML_POOL_DIR", os.path.join(R, "pools", "dclm-pool-400m-1x-html-jsonl-step3a-10pct"))
SRC = os.environ.get("FW_SRC", POOL + "/step3b_length_only/dclm_baseline_refinedweb_post_lang_length_only")
LID = os.environ.get("LID_MODEL", os.path.join(os.environ.get("DCLM_DIR", ""),
                     "baselines/mappers/enrichers/language_id_enrichment_models/lid.176.bin"))
OUT = sys.argv[1]; NP = int(sys.argv[2])
SH = os.path.join(OUT, "shards"); os.makedirs(SH, exist_ok=True)

from datatrove.data import Document
from datatrove.pipeline.filters import (URLFilter, LanguageFilter, GopherRepetitionFilter, GopherQualityFilter,
                                        C4QualityFilter, FineWebQualityFilter)


class TO(Exception):
    pass


def build():
    from fasttext.FastText import _FastText
    lf = LanguageFilter(); lf._model = _FastText(LID)
    uf = URLFilter(); uf.download_data()
    return uf, [("lang", lf), ("gopher_rep", GopherRepetitionFilter()), ("gopher_qual", GopherQualityFilter()),
                ("c4", C4QualityFilter(filter_no_terminal_punct=False)), ("fineweb", FineWebQualityFilter(exclusion_writer=None))]


UF = FILTERS = None


def _init():
    global UF, FILTERS
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))
    UF, FILTERS = build()


def extract(html, sec=60):
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.html import HTMLTree
    signal.alarm(sec)
    try:
        return extract_plain_text(HTMLTree.parse(html), main_content=True, alt_texts=False, preserve_formatting=True)
    except Exception:
        return None
    finally:
        signal.alarm(0)


def res(r):
    if isinstance(r, tuple):
        return bool(r[0]), r[1]
    return bool(r), None


def url_of(d):
    m = d.get("metadata")
    if isinstance(m, str):
        import ast
        try:
            m = ast.literal_eval(m)
        except Exception:
            m = {}
    return (m or {}).get("WARC-Target-URI") or d.get("url")


def one(stem):
    c = collections.Counter(); kept = []
    with gzip.open(os.path.join(SRC, stem + ".jsonl.gz"), "rt", encoding="utf-8") as g:
        for i, line in enumerate(g):
            d = json.loads(line); c["html_rows"] += 1
            url = url_of(d)
            if url:
                ok, why = res(UF.filter(Document(text="", id=str(i), metadata={"url": url})))
                if not ok:
                    c["drop_url_" + str(why)] += 1; continue
            else:
                c["no_url"] += 1
            t = extract(d.get("text") or "")
            if t is None:
                c["extract_failed"] += 1; continue
            if not t.strip():
                c["extract_empty"] += 1; continue
            c["extracted"] += 1
            if not res(FILTERS[1][1].filter(Document(text=t, id=str(i), metadata={})))[0]:
                c["diag_raw_gopher_rep_drop"] += 1
            t = "\n".join(l for l in t.splitlines() if l.strip())
            doc = Document(text=t, id=str(i), metadata={})
            for name, f in FILTERS:
                try:
                    ok, why = res(f.filter(doc))
                except Exception as e:
                    ok, why = False, "error_" + type(e).__name__
                if not ok:
                    c["drop_%s_%s" % (name, why)] += 1; break
            else:
                if doc.text.strip():
                    c["kept"] += 1; kept.append(doc.text)
                else:
                    c["drop_empty_after_c4"] += 1
    tmp = os.path.join(SH, stem + ".jsonl.gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fo:
        for t in kept:
            fo.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    os.replace(tmp, os.path.join(SH, stem + ".jsonl.gz"))
    return stem, dict(c)


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".jsonl.gz") or f.endswith(".jsonl"))
    random.Random(3).shuffle(files)
    stems = [f[: -len(".jsonl.gz")] for f in files[:60]]
    if os.environ.get("FW_TEST"):
        stems = stems[: int(os.environ["FW_TEST"])]
    build()  # extract url blacklists once before forking
    counts = {}
    with Pool(NP, initializer=_init) as p:
        for k, (s, c) in enumerate(p.imap_unordered(one, stems)):
            counts[s] = c; print("%2d/%d %s %s" % (k + 1, len(stems), s, c), flush=True)
    tot = collections.Counter()
    for c in counts.values():
        tot.update(c)
    docs = []
    for s in stems:  # shuffled-shard order
        rows = [json.loads(l)["text"] for l in gzip.open(os.path.join(SH, s + ".jsonl.gz"), "rt", encoding="utf-8")]
        rows = [t for t in rows if t.strip()]; random.Random(4).shuffle(rows); docs += rows[:100]
    tmp = os.path.join(OUT, "fineweb_rule_5k.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fo:
        for k, t in enumerate(docs[:5000]):
            fo.write(json.dumps({"row": k, "text": t}, ensure_ascii=False) + "\n")
    os.replace(tmp, os.path.join(OUT, "fineweb_rule_5k.jsonl"))
    json.dump({"total": dict(tot), "survival_rate_of_extracted": tot["kept"] / max(tot["extracted"], 1),
               "survival_rate_of_html": tot["kept"] / max(tot["html_rows"], 1),
               "sample_n": min(5000, len(docs)), "stems": stems, "per_shard": counts},
              open(os.path.join(OUT, "fw_counts.json"), "w"), indent=1)
    print("TOTAL", dict(tot)); print("SAMPLE", min(5000, len(docs)))
    print("FW_RULE_OK", flush=True)
