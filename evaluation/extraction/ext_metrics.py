r"""Extraction vs Dripper (Figure 8, left). usage: ext_metrics.py <page_table> <student_outputs> <baselines.jsonl> <out_dir> [resiliparse_fill.jsonl]
Systems: resiliparse (page table field = the pool extraction the baselines used; where it is null, the re-extraction
with the same settings from resiliparse_fill.py, which reproduces the pool text on 960/960 validation pages), trafilatura, justext (baselines.jsonl
from extract_baselines.py), student (pre-op extraction; derived from the raw program when the file has no extraction).
Reference: Dripper text (page table `drip`). Extra reference row: the full rendered page (no extraction at all).
Per page and system:
  token P/R/F1 = multiset overlap of \w+ tokens of the lowercased text, the tokenisation and the P/R/F1 formula of
  the fidelity-by-length analysis (analysis/analyze_fidelity_by_length_5k.py; empty system -> P=1, empty reference -> R=1; F1=0 if P+R=0);
  line P/R/F1 = same formula on the multiset of lines, a line = lowercase, runs of non-word chars -> one space, empty
  lines dropped (so bullets / dashes / pipes / whitespace do not matter, wording and segmentation do);
  empty = no non-whitespace output (None / failed extraction counts as empty and is also counted in `failed`).
Summary (ext_vs_dripper_<N>.json): per system macro means over pages + pooled (micro) P/R/F1 + empty share, each with a
percentile bootstrap 95% CI over pages (B=1000, seed 0), and paired student-minus-baseline differences of macro F1.
Also writes ext_texts_<N>.jsonl (the texts the judge reads) and ext_vs_dripper_<N>.perpage.jsonl.
"""
import sys, os, json, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from common import *  # noqa
import numpy as np
SYSTEMS = ["student", "resiliparse", "trafilatura", "justext"]
B = int(os.environ.get("BOOT", "1000"))

def main():
    pages = load_pages(sys.argv[1]); stu, src = load_student(sys.argv[2], pages)
    bl = {int(r["gid"]): r for r in read_jsonl(sys.argv[3])}; out = sys.argv[4]; os.makedirs(out, exist_ok=True)
    fill = {int(r["gid"]): r["resiliparse_refill"] for r in read_jsonl(sys.argv[5])} if len(sys.argv) > 5 else {}
    n_fill = 0
    N = len(pages); miss = [p["gid"] for p in pages if p["gid"] not in stu or p["gid"] not in bl]
    assert not miss, "pages without student output or baseline extraction: %d (first %s)" % (len(miss), miss[:5])
    texts, per = [], []
    failed = collections.Counter()
    for p in pages:
        g = p["gid"]; b = bl[g]
        t = {"gid": g, "rendered": p["rendered"], "dripper": p["drip"] or "", "student": stu[g]["extraction"],
             "resiliparse": p.get("resiliparse") if p.get("resiliparse") is not None else fill.get(g), "trafilatura": b["trafilatura"] if not b.get("err_trafilatura") else None,
             "justext": b["justext"] if not b.get("err_justext") else None}
        n_fill += p.get("resiliparse") is None and fill.get(g) is not None
        for s in SYSTEMS:
            if t[s] is None: failed[s] += 1; t[s] = ""
        texts.append(t)
        rt, rl = toks(t["dripper"]), line_bag(t["dripper"])
        row = {"gid": g, "teacher_op": p["op"], "student_op": stu[g]["decision"], "dripper_tokens": sum(rt.values())}
        for s in SYSTEMS + ["rendered"]:
            st, sl = toks(t[s]), line_bag(t[s])
            a = prf(*overlap(st, rt)); c = prf(*overlap(sl, rl))
            ov, ns, nr = overlap(st, rt); lov, lns, lnr = overlap(sl, rl)
            row[s] = {"tokP": a[0], "tokR": a[1], "tokF1": a[2], "lineP": c[0], "lineR": c[1], "lineF1": c[2], "empty": is_empty(t[s]),
                      "tok_ov": ov, "tok_sys": ns, "tok_ref": nr, "line_ov": lov, "line_sys": lns, "line_ref": lnr}
        per.append(row)
    write_jsonl_atomic("%s/ext_texts_%d.jsonl" % (out, N), texts)
    write_jsonl_atomic("%s/ext_vs_dripper_%d.perpage.jsonl" % (out, N), per)
    rng = np.random.default_rng(0); IDX = [rng.integers(0, N, N) for _ in range(B)]
    def ci(vec_fn):
        est = vec_fn(np.arange(N)); bs = np.array([vec_fn(ix) for ix in IDX]); lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"est": round(float(est), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
    summ = {"pages": N, "page_table": os.path.abspath(sys.argv[1]), "student_outputs": os.path.abspath(sys.argv[2]),
            "baselines": os.path.abspath(sys.argv[3]), "student_field_sources": src, "failed_extractions": dict(failed),
            "resiliparse_null_in_table_filled_from_reextraction": n_fill,
            "reference": "Dripper text (page table field drip)",
            "tokenization": "re.compile(r'\\w+').findall(text.lower()), multiset overlap; = TOK / P-R-F1 of the fidelity-by-length analysis (empty system -> P=1, empty reference -> R=1)",
            "line_unit": "line = lowercase, runs of non-word characters (incl. _) -> single space, stripped; empty dropped; multiset overlap",
            "macro": "mean over pages of per-page values; pooled = sum overlap / sum sizes over pages",
            "ci": "percentile bootstrap over pages, B=%d, seed 0" % B, "systems": {}}
    for s in SYSTEMS + ["rendered"]:
        M = {k: np.array([r[s][k] for r in per], dtype=float) for k in ("tokP", "tokR", "tokF1", "lineP", "lineR", "lineF1", "empty",
                                                                    "tok_ov", "tok_sys", "tok_ref", "line_ov", "line_sys", "line_ref")}
        ne = M["empty"] == 0
        d = {k + "_macro": ci(lambda ix, k=k: M[k][ix].mean()) for k in ("tokP", "tokR", "tokF1", "lineP", "lineR", "lineF1")}
        d["empty_share"] = ci(lambda ix: M["empty"][ix].mean())
        for u in ("tok", "line"):
            pP = lambda ix, u=u: M[u + "_ov"][ix].sum() / max(1, M[u + "_sys"][ix].sum())
            pR = lambda ix, u=u: M[u + "_ov"][ix].sum() / max(1, M[u + "_ref"][ix].sum())
            d[u + "P_pooled"] = ci(pP); d[u + "R_pooled"] = ci(pR)
            d[u + "F1_pooled"] = ci(lambda ix, pP=pP, pR=pR: 2 * pP(ix) * pR(ix) / max(1e-12, pP(ix) + pR(ix)))
        d["tokP_macro_nonempty"] = round(float(M["tokP"][ne].mean()), 4) if ne.any() else None
        d["n_empty"] = int(M["empty"].sum()); d["n_failed"] = failed.get(s, 0)
        summ["systems"][s if s != "rendered" else "rendered_page_no_extraction(reference row)"] = d
    diffs = {}
    for s in ("resiliparse", "trafilatura", "justext"):
        for k in ("tokF1", "lineF1"):
            a = np.array([r["student"][k] for r in per]); b = np.array([r[s][k] for r in per])
            diffs["student-%s_%s_macro" % (s, k)] = ci(lambda ix, a=a, b=b: (a[ix] - b[ix]).mean())
    summ["paired_diffs"] = diffs
    write_json_atomic("%s/ext_vs_dripper_%d.json" % (out, N), summ)
    f = lambda x: "%.2f [%.2f, %.2f]" % (100 * x["est"], 100 * x["lo"], 100 * x["hi"])
    print("pages", N, "failed", dict(failed), "student fields", src)
    print("%-12s %-22s %-22s %-22s %-22s %-22s" % ("system", "tok P", "tok R", "tok F1 (macro)", "line F1 (macro)", "empty %"))
    for s, d in summ["systems"].items():
        print("%-12s %-22s %-22s %-22s %-22s %-22s" % (s[:12], f(d["tokP_macro"]), f(d["tokR_macro"]), f(d["tokF1_macro"]), f(d["lineF1_macro"]), f(d["empty_share"])))
    print("pooled tok F1:", {s[:12]: round(100 * d["tokF1_pooled"]["est"], 2) for s, d in summ["systems"].items()})
    print("EXT_METRICS_DONE")
if __name__ == "__main__":
    main()
