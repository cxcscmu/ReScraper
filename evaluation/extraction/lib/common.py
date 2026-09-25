"""Shared helpers for the extraction-quality metrics and the extraction judge.

Page table (heldout5k schema, one JSON object per line, keyed by gid):
  gid, stem, step1_idx, in_md5, url, warc_id, input (numbered "<lid:n> text" rendering the student reads), drip (Dripper
  text), output (teacher program), tag (teacher op), gfinal (teacher final text; "" for delete), edu, resiliparse (text or null)
HTML table: gid, html
Student outputs (one row per gid; field names are matched flexibly, see load_student):
  raw program, final text, decision, pre-op extraction (heldout_pipeline/work/<tag>/ours.jsonl: text / decision /
  extracted / status). Missing derived fields are recomputed from `raw` with lib/rescraper_ops.py (decision-first staged format).
"""
import json, os, re, sys, random, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "lib"))
import editops as E          # noqa: E402  (number_lines / apply_ops / OPS_RM / OPS_SUB)
import rescraper_ops as X          # noqa: E402  (parse_staged / body_from_prediction_dfirst)

# ---------------------------------------------------------------- text helpers
TOK = re.compile(r"\w+")     # = the \w+ tokenisation of the fidelity-by-length analysis, applied to .lower()
LID = re.compile(r"^<lid:\d+> ?")

def strip_lid(inp):
    """numbered rendering -> plain rendered page text (what a reader of the page sees, incl. site chrome)"""
    return "\n".join(LID.sub("", x) for x in (inp or "").split("\n"))

def toks(s):
    return collections.Counter(TOK.findall((s or "").lower()))

def norm_line(s):
    """line identity for line-level P/R/F1: lowercase, every run of non-word characters -> one space, stripped.
    Makes bullets / list dashes / table pipes / whitespace differences between extractors irrelevant."""
    return re.sub(r"[\W_]+", " ", s.lower()).strip()

def line_bag(s):
    return collections.Counter(x for x in (norm_line(l) for l in (s or "").split("\n")) if x)

def prf(ov, ns, nr):
    """the convention of the fidelity-by-length analysis: empty system -> P=1, empty reference -> R=1"""
    p = ov / ns if ns else 1.0
    r = ov / nr if nr else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def overlap(a, b):
    return sum((a & b).values()), sum(a.values()), sum(b.values())

def is_empty(s):
    return s is None or not str(s).strip()

# ---------------------------------------------------------------- ops / tags
OPN = {"<extract>": "keep", "<keep>": "keep", "keep": "keep", "extract": "keep",
       "<refine>": "edit", "<edit>": "edit", "edit": "edit", "refine": "edit",
       "<delete>": "delete", "delete": "delete", "<rewrite>": "rewrite", "rewrite": "rewrite"}

def op(tag):
    return OPN.get((tag or "").strip().lower(), (tag or "").strip() or "none")

DEC_LEAD = {"<keep>", "<edit>", "<delete>", "<rewrite>", "<clean>"}

def student_extraction(raw, ninp):
    """pre-op (stage-1) extraction of ReScraper: apply the <extract> block's line removals to the
    numbered input. Handles the decision-first format (line 1 = decision tag). None if there is no <extract> block."""
    ls = (raw or "").strip().split("\n")
    if ls and ls[0].strip() in DEC_LEAD and len(ls) > 1:
        ls = ls[1:]
    if not ls or ls[0].strip() != "<extract>":
        return None
    ops1, _, _ = X.parse_staged("\n".join(ls))
    return E.apply_ops(ninp, ops1)

def student_final(raw, ninp):
    """(teacher-style tag, final text) exactly as the evaluation reads a decision-first prediction"""
    return X.body_from_prediction_dfirst(ninp, raw or "")

# ---------------------------------------------------------------- IO
def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl_atomic(p, rows):
    tmp = p + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)

def write_json_atomic(p, obj):
    tmp = p + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, p)

def load_pages(path):
    rows = read_jsonl(path)
    for r in rows:
        r["gid"] = int(r["gid"])
        r["rendered"] = strip_lid(r["input"])
        r["op"] = op(r.get("tag") or X.parse_staged(r.get("output") or "")[1])
        r["gfinal"] = r.get("gfinal") or ""
    rows.sort(key=lambda r: r["gid"])
    assert len({r["gid"] for r in rows}) == len(rows), "duplicate gid in page table"
    return rows

def _first(d, names):
    for n in names:
        if n in d and d[n] is not None:
            return d[n], n
    return None, None

def load_student(path, pages):
    """-> {gid: {raw, final, decision (keep/edit/delete/rewrite), extraction (str or None), src: which field each came from}}"""
    byg = {p["gid"]: p for p in pages}
    out = {}; src = collections.Counter()
    for d in read_jsonl(path):
        g = int(d["gid"]); p = byg.get(g)
        if p is None:
            continue
        raw, _ = _first(d, ["raw", "program", "raw_program", "raw_output", "pred", "prediction"])
        fin, fn = _first(d, ["final", "final_text", "pfinal", "text"])
        dec, dn = _first(d, ["decision", "op", "pt", "tag"])
        ext, en = _first(d, ["extraction", "extracted", "pre_op_extraction", "preop", "stage1", "extract", "stage1_text"])
        if d.get("status") not in (None, "ok"):
            # page the student never produced a program for (e.g. prompt over the context limit): the corpus gets
            # nothing, so final text "" and decision "omitted"; extraction unavailable (None -> counted as failed/empty)
            src["status:" + str(d.get("status"))] += 1
            out[g] = {"raw": raw, "final": "", "decision": "omitted", "extraction": None}
            continue
        if raw is not None and (fin is None or dec is None):
            t, body = student_final(raw, p["input"])
            if fin is None: fin, fn = body, "raw->body_from_prediction_dfirst"
            if dec is None: dec, dn = t, "raw->body_from_prediction_dfirst"
        if ext is None and raw is not None:
            ext, en = student_extraction(raw, p["input"]), "raw->apply_ops(input, stage-1 ops)"
        src["final:" + str(fn)] += 1; src["decision:" + str(dn)] += 1; src["extraction:" + str(en)] += 1
        out[g] = {"raw": raw, "final": fin or "", "decision": op(dec), "extraction": ext}
    return out, dict(src)

# ---------------------------------------------------------------- stats
def bootstrap_ci(values_by_page, stat, B=1000, seed=0):
    """values_by_page: list of per-page tuples; stat(list_of_indices) -> float. Percentile 95% CI."""
    import numpy as np
    rng = np.random.default_rng(seed); n = len(values_by_page)
    est = stat(np.arange(n))
    bs = [stat(rng.integers(0, n, n)) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(est), float(lo), float(hi)
