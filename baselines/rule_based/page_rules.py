"""Lang+length gate and the DCLM rule stacks (RefinedWeb-rule, C4-rule) on the resiliparse text of individual pages,
applied page by page with DCLM's own mapper factory (baselines.core.factories.get_mapper, _safe=True, exactly as
process_single_file does: an exception = the page is dropped).

Gate (a page failing it is deleted for every baseline):
  dclm_baseline_refinedweb_lang_only.yaml  (URL filters, url_removal, newline_removal, fastText en >= 0.65)
  -> dclm_baseline_refinedweb_post_lang_length_only.yaml  (page_length_filter: 50..100000 words, ignore punct.)
  i.e. the two configs that define the pool dclm_baseline_refinedweb_post_lang_length_only. The text handed
  downstream is the raw resiliparse text.
  refinedweb_banned_domains_curated.txt is not shipped with the DCLM repository, so that one URL filter is skipped;
  every page of the source pool already passed DCLM's URL + language step upstream.
  The authoritative gate is the input field `in_langlen_pool` (the page's HTML is in the lang+length-filtered pool);
  the recomputed gate is written alongside as a diagnostic.
RefinedWeb-rule: dclm_baseline_refinedweb_post_lang.yaml, as the 7.04B RefinedWeb-rule corpus
  (run_resiliparse_pipeline.sh: resiliparse_extract -> post_lang -> bff dedup).
C4-rule: DCLM's c4.yaml with its two corpus-level exact_dedup steps (url, paragraph) removed; all per-page steps kept,
  including within_page_dedup (a per-page line dedup) and the langdetect en >= 0.99 filter.
Input rows: {"k", "resiliparse" (text or null), "metadata" (WARC metadata, dict or str), "url", "in_langlen_pool",
  "in_step3a"}. Output rows: {"k", "gate", "gate_reason", "rw", "c4", "rw_reason", "c4_reason", diagnostics}.
usage (cwd $DCLM_DIR, DCLM data-processing env): page_rules.py <extracted.jsonl> <out.jsonl>
"""
import sys, json, copy, collections, yaml, os
DCLM = os.environ.get("DCLM_DIR", os.getcwd())
sys.path.insert(0, DCLM)
from baselines.core.factories import get_mapper
# nltk >= 3.9 refuses the pickled punkt model that core_utils._prep_nltk_tokenizer loads ("english.pickle"): the
# LookupError is swallowed and sent_tokenizer stays None, so every C4 sentence-length check errors. Use the same
# English Punkt parameters in nltk's punkt_tab format; core_utils only reads the module global sent_tokenizer.
import baselines.mappers.core_utils as _cu
from nltk.tokenize.punkt import PunktTokenizer
_cu.sent_tokenizer = PunktTokenizer("english")
CFG = os.path.join(DCLM, "baselines/baselines_configs/")
TLDS = os.path.join(DCLM, "baselines/mappers/iana_tlds.txt")

def steps_of(name, drop=()):
    st = yaml.safe_load(open(CFG + name))[0]["steps"]
    out = []
    for s in st:
        s = dict(s); s.pop("_aggregate", None)
        if s["func"] in drop: continue
        if s.get("banlist_from_fname") and not os.path.exists(os.path.join(DCLM, s["banlist_from_fname"])):
            print("SKIP step (banlist file not shipped with the DCLM repo):", name, s["func"], s["banlist_from_fname"]); continue
        if s["func"] == "url_removal_modifier": s.setdefault("tlds_filepath", TLDS)  # the default path is not repo-relative
        out.append(s)
    return out

def build(steps):
    return [(s["func"], get_mapper(**s, _safe=True)) for s in steps]

def run(stack, page):
    pages = [page]
    for name, f in stack:
        nxt = []
        for p in pages:
            r = f(p)
            if isinstance(r, list): nxt.extend(r)
            else: return [], name + " ERROR " + str(r)[:200]
        pages = nxt
        if not pages: return [], name
    return pages, None

LANG = build(steps_of("dclm_baseline_refinedweb_lang_only.yaml"))
LEN = build(steps_of("dclm_baseline_refinedweb_post_lang_length_only.yaml"))
RW = build(steps_of("dclm_baseline_refinedweb_post_lang.yaml"))
C4 = build(steps_of("c4.yaml", drop=("exact_dedup",)))
print("RW steps:", [n for n, _ in RW]); print("C4 steps:", [n for n, _ in C4]); print("LANG steps:", [n for n, _ in LANG])

def page_of(r, text):
    m = r.get("metadata")
    if isinstance(m, str):
        import ast
        try: m = ast.literal_eval(m)
        except Exception: m = {}
    m = dict(m or {})
    if r.get("url") and "WARC-Target-URI" not in m: m["WARC-Target-URI"] = r["url"]
    return {"text": text, "metadata": m}

S = collections.Counter(); why = collections.defaultdict(collections.Counter)
with open(sys.argv[2], "w", encoding="utf-8") as fo:
    for l in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(l); t = r.get("resiliparse")
        o = {"k": r["k"], "gate": None, "gate_reason": None, "rw": "", "c4": "", "rw_reason": None, "c4_reason": None}
        if t is None or not t.strip():
            o["gate"] = False; o["gate_reason"] = "extract_none" if t is None else "extract_empty"
        else:
            lp, lr = run(LANG, page_of(r, t))
            o["lang_pass"] = bool(lp); o["lang_reason"] = lr
            ln, nr = run(LEN, lp[0]) if lp else ([], None)
            o["len_pass_after_lang"] = bool(ln)
            raw_len, rr = run(LEN, page_of(r, t)); o["len_pass_raw"] = bool(raw_len)
            o["gate_recomputed"] = bool(lp) and bool(ln); o["gate_recomputed_reason"] = lr or nr
            # authoritative gate = the actual output of DCLM's lang+length step: the page's HTML is in the
            # dclm_baseline_refinedweb_post_lang_length_only pool (the recomputation above is a diagnostic)
            o["gate"] = bool(r.get("in_langlen_pool")); o["gate_reason"] = None if o["gate"] else "not_in_langlen_pool"
        o["in_langlen_pool"] = r.get("in_langlen_pool"); o["in_step3a"] = r.get("in_step3a")
        if o["gate"]:
            p, why_rw = run(RW, page_of(r, t)); o["rw"] = "\n".join(x["text"] for x in p) if p else ""; o["rw_reason"] = why_rw
            p, why_c4 = run(C4, page_of(r, t)); o["c4"] = "\n".join(x["text"] if isinstance(x["text"], str) else "\n".join(x["text"]) for x in p) if p else ""; o["c4_reason"] = why_c4
            S["rw_kept"] += bool(o["rw"].strip()); S["c4_kept"] += bool(o["c4"].strip())
            why["rw"][why_rw or "kept"] += 1; why["c4"][why_c4 or "kept"] += 1
        S["pages"] += 1; S["gate_pass"] += bool(o["gate"]); why["gate"][o["gate_reason"] or "pass"] += 1
        S["recomputed_gate_pass"] += bool(o.get("gate_recomputed")); why["gate_recomputed"][o.get("gate_recomputed_reason") or "pass"] += 1
        fo.write(json.dumps(o, ensure_ascii=False) + "\n")
print(dict(S))
for k, v in why.items(): print(k, dict(v.most_common()))
print("RULES_DONE")
