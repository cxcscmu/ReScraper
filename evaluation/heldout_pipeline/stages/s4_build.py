"""Stage 4 (CPU): attach scores to gids and write the figure data files (schemas of the analysis/data files).

Inputs: <work>/{ours,sys_ultrax,sys_proxc,sys_refinedweb_rule,sys_fineweb_rule}.jsonl, ux_out.jsonl,
        texts_index.jsonl + score_in/ + score_out/ (DataMan/Edu) (+ score_cache.jsonl for the release-decoding run)
Outputs (heldout5k -> *_5k.json in $HELDOUT_PIPELINE_DIR/data; any other table -> *_<tag>.json in <work>/data; HP_OUT / HP_SUF
override; the release-decoding chain writes *_5k_rel.json):
  scores_by_gid_<suf>.jsonl      per page: pre / post[system] / ours_extracted DataMan + FineWeb-Edu
  operation_scores_<suf>.json    groups keep / delete / edit_before / edit_after / rewrite_before / rewrite_after
  operation_mix_<suf>.json       page-level operation mix of ReScraper, UltraX, ProX-C (+ per-page bins)
  quality_buckets_<suf>.json     pre / post DataMan + FineWeb-Edu per page and pipeline
  dist_length_<suf>.json         GPT-NeoX-20B token and char lengths of every kept output
  build_report_<suf>.json        counts and asserts
HP_DECODING = greedy (default) | release: only changes the description strings.
usage: s4_build.py <page_table>
"""
import collections, difflib, itertools, json, math, os, random, re, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from common import *
X, E = e2e()
PT = sys.argv[1]; W = workdir(PT); TAG = tag_of(PT)
SUF = os.environ.get("HP_SUF") or ("5k" if TAG == "heldout5k" else TAG)
OUT = os.environ.get("HP_OUT") or (HP + "/data" if TAG == "heldout5k" else W + "/data"); os.makedirs(OUT, exist_ok=True)
DECODING = {"greedy": "greedy run of the released checkpoint (T=0, max_tokens<=8192, stages/s1_ours_infer.py)",
            "release": "release decoding of the released checkpoint (T=1.0, top_p=1.0, max_tokens=3072; operation-ablation "
                       "full-arm records, stages/s1_rel_exec.py)"}[os.environ.get("HP_DECODING", "greedy")]
NOW = time.strftime("%Y-%m-%d %H:%M %Z")
REPORT = {"page_table": PT, "tag": TAG, "created": NOW}
norm = lambda t: " ".join((t or "").split()); nw = lambda t: len((t or "").split())

pages = load_pages(PT, ["gid", "resiliparse", "input"])
N = len(pages)
ours = load_by_gid(W + "/ours.jsonl")
SYSF = {"ultrax": "sys_ultrax", "proxc": "sys_proxc", "refinedweb_rule": "sys_refinedweb_rule", "fineweb_rule": "sys_fineweb_rule"}
SYS = {"rescraper": ours}
for k, f in SYSF.items():
    SYS[k] = load_by_gid(W + "/%s.jsonl" % f)
for k, v in SYS.items():
    assert sorted(v) == list(range(N)), k
text_of = lambda k, g: (SYS[k][g].get("text") if nonempty(SYS[k][g].get("text")) else None)

# ------------------------------------------------------------------ A. scores by unique text
dm, edu = {}, {}
nin = 0
if os.path.exists(W + "/score_cache.jsonl"):     # scores reused by exact text hash (s3_reuse.py)
    for r in read_jsonl(W + "/score_cache.jsonl"):
        dm[r["h"]] = r["dataman"]; edu[r["h"]] = r["fineweb_edu"]
    REPORT["scores_from_cache"] = len(dm)
for f in sorted(os.listdir(W + "/score_in")):
    b = f[:-len(".jsonl")]
    ins = list(read_jsonl(W + "/score_in/" + f)); nin += len(ins)
    assert all(nonempty(r["text"]) for r in ins)
    d = list(read_jsonl("%s/score_out/dataman_%s/%s.scores.jsonl" % (W, b, b)))
    e = list(read_jsonl("%s/score_out/edu_%s/%s.edu.jsonl" % (W, b, b)))
    assert len(d) == len(ins) == len(e), ("scorer output count != input count", b, len(ins), len(d), len(e))
    for r, x, y in zip(ins, d, e):
        o = x.get("overall_score")
        dm[r["h"]] = float(o) if isinstance(o, float) and 1.0 <= o <= 5.0 else None
        edu[r["h"]] = float(y["edu"])
REPORT["unique_texts_scored"] = nin
need = {x["h"] for x in read_jsonl(W + "/texts_index.jsonl")}
assert need <= set(dm), ("texts without a score", len(need - set(dm)))
REPORT["dataman_invalid"] = sum(v is None for v in dm.values())
idx = collections.defaultdict(dict)
for x in read_jsonl(W + "/texts_index.jsonl"):
    idx[x["gid"]][x["field"]] = x["h"]
def sc(h):
    return None if h is None else {"dataman": dm[h], "fineweb_edu": edu[h]}
rows = []
for p in pages:
    g = p["gid"]; ix = idx.get(g, {})
    rows.append({"gid": g, "pre": sc(ix.get("pre")),
                 "post": {k: sc(ix.get("post:" + k)) for k in SYS},
                 "ours_extracted": sc(ix.get("ours_extracted")),
                 "status": {k: SYS[k][g].get("status") for k in SYS}})
    for k in SYS:     # a kept text must have been scored, a missing text must not
        assert (rows[-1]["post"][k] is None) == (text_of(k, g) is None), (g, k)
write_jsonl_atomic(OUT + "/scores_by_gid_%s.jsonl" % SUF, rows)

# ------------------------------------------------------------------ B. operation scores
NAME = {"<keep>": "keep", "<delete>": "delete", "<edit>": "edit", "<rewrite>": "rewrite"}
G = collections.defaultdict(lambda: {"dataman": [], "fineweb_edu": [], "gid": []}); chk = collections.Counter()
def put(grp, g, t):
    if not t: chk[grp + "_empty_dropped"] += 1; return      # only non-empty texts are scored
    h = md5(t); d = dm[h]
    if d is None: chk[grp + "_dataman_invalid_dropped"] += 1; return
    G[grp]["dataman"].append(d); G[grp]["fineweb_edu"].append(edu[h]); G[grp]["gid"].append(g)
for p in pages:
    o = ours[p["gid"]]
    if o["status"] != "ok" or o["decision"] not in NAME:
        chk["no_decision_line"] += 1; continue
    before = (o["extracted"] or "").strip(); after = (o["text"] or "").strip(); op = NAME[o["decision"]]
    chk[op] += 1
    if op == "keep":
        chk["keep_before==after"] += before == after; put("keep", p["gid"], before)
    elif op == "delete":
        put("delete", p["gid"], before)
    else:
        put(op + "_before", p["gid"], before); put(op + "_after", p["gid"], after)
OPS = {"description": ("DataMan-1.5B-EN overall score and FineWeb-Edu score per document, for the %d pages of the %s page table "
                       "grouped by the operation ReScraper chose (%s). "
                       "before = page after the model's own <extract> removals; after = executor output; keep has before == after. "
                       "Empty texts dropped; 'gid' lists the page of each score." % (N, TAG, DECODING)),
       "source": "evaluation/heldout_pipeline (s1 student outputs, scorers/score_dataman.py + score_edu.py, stages/s4_build.py)",
       "created": NOW, "counts": dict(chk),
       "groups": {k: G[k] for k in ("keep", "delete", "edit_before", "edit_after", "rewrite_before", "rewrite_after")}}
json.dump(OPS, open(OUT + "/operation_scores_%s.json" % SUF, "w"), indent=1)
REPORT["operation_scores"] = {k: len(v["dataman"]) for k, v in OPS["groups"].items()} | {"checks": dict(chk)}

# ------------------------------------------------------------------ C. operation mix
def effect_split(src, out):
    a = [norm(l) for l in (src or "").split("\n") if norm(l)]; b = [norm(l) for l in (out or "").split("\n") if norm(l)]
    if not b: return "emptied"
    if a == b: return "unchanged"
    dl = st = False
    for t, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if t == "equal": continue
        if t == "delete": dl = True
        elif t == "replace" and (i2 - i1) == (j2 - j1): st = True
        else: dl = dl or (i2 - i1) > (j2 - j1); st = True
    return "lines + strings" if dl and st else "lines only" if dl else "strings only"
EDGES = (0.10, 0.50)
BINS = ["keep", "edit_lt10", "edit_10_50", "edit_gt50", "delete", "rewrite", "not_processed"]
def ebin(f): return "edit_lt10" if f < EDGES[0] else "edit_10_50" if f <= EDGES[1] else "edit_gt50"
def classify(cls, src, out):
    if cls == "omitted": return "not_processed", None
    if cls in ("delete", "rewrite", "keep"): return cls, None
    e = effect_split(src, out)
    if e == "unchanged": return "keep", None
    if e == "emptied": return "delete", None
    f = max(0.0, 1 - nw(out) / max(nw(src), 1)); return ebin(f), f
CALL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
def classify_ux(o, prog, cleaned):  # UltraX program classes, class only
    s = set(CALL.findall(prog or "")); no, nc = norm(o), norm(cleaned)
    if not nc: return "delete"
    if "add_line" in s: return "other: add_line"
    if nc == no: return "keep"
    e = s & {"remove_lines", "replace_str"}
    if e == {"remove_lines"}: return "edit: lines only"
    if e == {"replace_str"}: return "edit: strings only"
    if e == {"remove_lines", "replace_str"}: return "edit: lines + strings"
    return "other: text changed without remove_lines/replace_str"
uxo = load_by_gid(W + "/ux_out.jsonl")
res = {"ReScraper": [], "UltraX": [], "ProX-C": []}; per_page = []
for p in pages:
    g = p["gid"]; orig = p["resiliparse"] or ""
    o = ours[g]
    res["ReScraper"].append(classify(o["opmix_cls"], o.get("extracted"), o.get("text") if o.get("emits") else ""))
    u = SYS["ultrax"][g]
    if u["status"] in ("no_input", "not_found"):
        res["UltraX"].append(("not_processed", None))
    else:
        cl = uxo[g].get("cleaned") or ""
        c = classify_ux(orig, uxo[g].get("model_output"), cl)
        c = "delete" if not norm(cl) else "keep" if c == "keep" else "edit"
        res["UltraX"].append(classify(c, orig, cl))
    r = SYS["proxc"][g]
    if r["status"] == "no_input":
        res["ProX-C"].append(("not_processed", None))
    else:
        t = (r.get("text") or "") if r["status"] not in ("page blanked", "execute failed") else ""
        c = {"unchanged": "keep", "page blanked": "delete"}.get(r["status"], "edit")
        res["ProX-C"].append(classify(c, orig, t))
    per_page.append({"gid": g, **{k: {"bin": res[k][-1][0], "removed": res[k][-1][1]} for k in res}})
MIX = {"description": ("Page-level operation mix of the three model-based pipelines on the same %d pages of the %s page table: "
                       "UltraX and ProX-C read the page's pool resiliparse_extract text; ReScraper reads its own <lid:n> rendering. "
                       "Every page is classified by what changes in the output text, with one rule for all systems: "
                       "keep = same non-empty whitespace-normalized lines as the input, delete = empty output, rewrite = ReScraper's "
                       "<rewrite> branch, and every other page is an edit, binned by the share of the input's words it removes "
                       "(1 - words out / words in; below 10%%, 10-50%%, above 50%%)." % (N, TAG)),
       "input_note": ("UltraX output = the page's row of the UltraX run over the raw resiliparse pool (post/inf parquets, "
                      "joined by exact text); ProX-C = s2_proxc.py on the same text; ReScraper (%s) classes compare its "
                      "output with the text left after its first-stage <extract> removals." % DECODING),
       "source": "evaluation/heldout_pipeline/stages/s4_build.py (per-page bins in operation_mix_perpage_%s.jsonl)" % SUF,
       "operations": BINS, "edges_removed_word_share": list(EDGES), "pages": N, "systems": {}}
for k, v in res.items():
    c = collections.Counter(b for b, _ in v); assert sum(c.values()) == N
    fr = sorted(f for b, f in v if f is not None)
    MIX["systems"][k] = {"counts": {b: c[b] for b in BINS}, "edited_pages": len(fr),
                         "removed_share_quantiles_edited": {str(q): round(fr[int(q * len(fr))], 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if fr else {}}
nr = collections.Counter(ours[g]["status"] for g in range(N) if res["ReScraper"][g][0] == "not_processed")
MIX["systems"]["ReScraper"]["not_processed_note"] = "; ".join("%d %s" % (v, k) for k, v in nr.items()) or "none"
json.dump(MIX, open(OUT + "/operation_mix_%s.json" % SUF, "w"), indent=1)
write_jsonl_atomic(OUT + "/operation_mix_perpage_%s.jsonl" % SUF, per_page)
REPORT["operation_mix"] = {k: v["counts"] for k, v in MIX["systems"].items()}

# ------------------------------------------------------------------ D. quality buckets
ORDER = ["refinedweb_rule", "fineweb_rule", "proxc", "ultrax", "rescraper"]
qb_pages = []
for r in rows:
    pre = r["pre"] or {"dataman": None, "fineweb_edu": None}
    qb_pages.append({"gid": r["gid"], "pre": {"dataman": pre["dataman"], "fineweb_edu": pre["fineweb_edu"]},
                     "post": {k: (None if r["post"][k] is None else r["post"][k]) for k in ORDER}})
bad = [(q["gid"], k) for q in qb_pages for k in ORDER if q["post"][k] is not None and q["post"][k]["dataman"] is None]
REPORT["quality_buckets_post_dataman_invalid"] = len(bad)
for g, k in bad:           # the plot averages post scores of kept pages; an unparseable DataMan output cannot be averaged
    qb_pages[g]["post"][k] = dict(qb_pages[g]["post"][k], dataman=float("nan"))
QB = {"description": ("Per page of the %s page table: DataMan (1-5) and FineWeb-Edu of the page's resiliparse text (pre) and of "
                      "each pipeline's output (post; null = the pipeline deleted the page or could not process it)." % TAG),
      "created": NOW, "order": ORDER, "pages": qb_pages}
json.dump(QB, open(OUT + "/quality_buckets_%s.json" % SUF, "w"))

# ------------------------------------------------------------------ E. kept texts per system (gid order)
KEYS = ["rescraper", "ultrax", "proxc", "refinedweb_rule", "fineweb_rule"]
kept = {k: [(g, text_of(k, g)) for g in range(N) if text_of(k, g) is not None] for k in KEYS}
kept["resiliparse"] = [(p["gid"], p["resiliparse"]) for p in pages if nonempty(p["resiliparse"])]
REPORT["kept_docs"] = {k: len(v) for k, v in kept.items()}
SRC = {"rescraper": "released student (%s), executed with lib/e2e_ops.py" % DECODING,
       "ultrax": "UltraX run over the raw resiliparse pool (post/<stem>.parquet `cleaned`), row joined by exact text",
       "proxc": "gair-prox/web-chunk-refining-lm, settings of the pool ProX-C run (s2_proxc.py)",
       "refinedweb_rule": "DCLM dclm_baseline_refinedweb_post_lang.yaml mappers (s2_rw_rule.py)",
       "fineweb_rule": "FineWeb-rule chain, datatrove 0.2.0 (s2_fw_rule.py)",
       "resiliparse": "pool resiliparse_extract text of the page"}
SRC = {k: v + "; page table %s, kept (non-empty) outputs only, no dedup" % TAG for k, v in SRC.items()}

# ------------------------------------------------------------------ F. length distribution (GPT-NeoX-20B tokens)
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b"); tk.model_max_length = 10 ** 12
TOKS = {}
def ntoks(k):
    if k not in TOKS:
        ids = tk([t for _, t in kept[k]], add_special_tokens=False, return_attention_mask=False)["input_ids"] if kept[k] else []
        TOKS[k] = {g: x for (g, _), x in zip(kept[k], ids)}
    return TOKS[k]
L = {"metric": "per-document length in GPT-NeoX-20B tokens (no special tokens, full document, no truncation)",
     "tokenizer": "EleutherAI/gpt-neox-20b", "sources": {}, "lengths": {}, "chars": {}, "summary": {}, "pending": []}
for k in KEYS + ["resiliparse"]:
    T = ntoks(k); lens = [len(T[g]) for g, _ in kept[k]]; a = np.array(lens, dtype=np.float64)
    L["sources"][k] = SRC[k]; L["lengths"][k] = lens; L["chars"][k] = [len(t) for _, t in kept[k]]; L["gids"] = L.get("gids", {}); L["gids"][k] = [g for g, _ in kept[k]]
    L["summary"][k] = {"n": len(lens), "median": float(np.median(a)), "mean": float(a.mean()), "p10": float(np.percentile(a, 10)),
                       "p90": float(np.percentile(a, 90)), "min": int(a.min()), "max": int(a.max()), "zero_len": int((a == 0).sum())}
json.dump(L, open(OUT + "/dist_length_%s.json" % SUF, "w"))
REPORT["length_median"] = {k: v["median"] for k, v in L["summary"].items()}
json.dump(REPORT, open(OUT + "/build_report_%s.json" % SUF, "w"), indent=1)
print(json.dumps(REPORT, indent=1)); print("S4_BUILD_OK", OUT)
