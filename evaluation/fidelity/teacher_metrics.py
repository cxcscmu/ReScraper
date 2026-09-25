"""Teacher fidelity of ReScraper on the held-out pages: decision accuracy, keep-or-delete accuracy, pooled token
P/R/F1 of the text each page contributes (deleted pages -> ""), 95% page-bootstrap CIs, the 4x4 teacher -> student
confusion, and the decision flow of the Sankey figure.

Conventions (as the held-out evaluation of the SFT runs):
  gt, gbody = e2e_ops.body_from_prediction_dfirst(input, teacher output)   (teacher tag: <extract>/<refine>/<delete>/<rewrite>)
  pt, pbody = the executed student program (same reader)
  decision accuracy over the pages sent to the model; token P/R/F1 = corpus-level overlap of lower-cased \\w+ token
  multisets of pfinal vs gfinal (deleted -> ""), on pages whose prompt + teacher target fit the 32,768-token context
  ("full"); CIs from 2,000 page-bootstrap resamples (random.Random(0)).
"""
import collections, os, random, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import e2e_ops as X  # noqa: E402

TOK = re.compile(r"\w+")
MAXLEN = 32768
TAGS = ("<extract>", "<refine>", "<delete>", "<rewrite>")


def teacher_metrics(rows, nboot=2000):
    """rows: (gt, pt, gfinal, pfinal, full) per page sent to the model."""
    def stats(sel):
        conf = collections.Counter((rows[i][0], rows[i][1]) for i in sel)
        n = len(sel); corr = sum(v for (g, p), v in conf.items() if g == p)
        ov = st = gtok = 0
        for i in sel:
            g, p, gf, pf, full = rows[i]
            if not full: continue
            pc, gc = collections.Counter(TOK.findall(pf.lower())), collections.Counter(TOK.findall(gf.lower()))
            ov += sum((pc & gc).values()); st += sum(pc.values()); gtok += sum(gc.values())
        P = ov / max(1, st); R = ov / max(1, gtok); F = 2 * P * R / max(1e-9, P + R)
        kd = sum(v for (g, p), v in conf.items() if (g == "<delete>") == (p == "<delete>"))
        return {"n": n, "acc": 100 * corr / max(1, n), "keep_or_delete_acc": 100 * kd / max(1, n),
                "tok_P": 100 * P, "tok_R": 100 * R, "tok_F1": 100 * F, "conf": conf}
    allsel = list(range(len(rows)))
    s = stats(allsel)
    # pre-tokenise once for the bootstrap
    pre = []
    for g, p, gf, pf, full in rows:
        pc, gc = collections.Counter(TOK.findall(pf.lower())), collections.Counter(TOK.findall(gf.lower()))
        pre.append((g == p, sum((pc & gc).values()) if full else 0, sum(pc.values()) if full else 0, sum(gc.values()) if full else 0))
    rng = random.Random(0); A, F = [], []
    for _ in range(nboot):
        sel = [rng.randrange(len(rows)) for _ in rows]
        A.append(100 * sum(pre[i][0] for i in sel) / len(sel))
        ov = sum(pre[i][1] for i in sel); st = sum(pre[i][2] for i in sel); gt = sum(pre[i][3] for i in sel)
        P, R = ov / max(1, st), ov / max(1, gt); F.append(100 * 2 * P * R / max(1e-9, P + R))
    A.sort(); F.sort()
    ci = lambda v: [round(v[int(0.025 * len(v))], 2), round(v[int(0.975 * len(v)) - 1], 2)]
    conf = s.pop("conf")
    s = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()}
    s.update(acc_ci95=ci(A), tok_F1_ci95=ci(F),
             confusion_gold_to_pred={g: {p: conf[(g, p)] for p in TAGS + ("<none>",) if conf[(g, p)]} for g in TAGS},
             pred_counts=dict(collections.Counter(r[1] for r in rows)), gold_counts=dict(collections.Counter(r[0] for r in rows)))
    return s


def rows_for(sel_pages, pred_of):
    """pred_of(gid) -> None (not sent) | (raw_program, full) | ((pt, pfinal), full) for dumps that kept no raw text."""
    rs = []
    for p in sel_pages:
        pr = pred_of(p["gid"])
        if pr is None: continue          # not sent to the model
        raw, full = pr
        gt, gb = X.body_from_prediction_dfirst(p["input"], p["output"])
        pt, pb = raw if isinstance(raw, tuple) else X.body_from_prediction_dfirst(p["input"], raw)
        rs.append((gt, pt, "" if gt == "<delete>" else gb, "" if pt == "<delete>" else pb, full))
    return rs


def rows_from_executed(pages, outs):
    """(gid, row) for teacher_metrics from executed outputs (ours.jsonl schema: status, pt, text, plen, tlen);
    pages the model never saw (status != ok) are left out."""
    rows = []
    for p, o in zip(pages, outs):
        if o["status"] != "ok" or not p.get("output"): continue
        gt, gb = X.body_from_prediction_dfirst(p["input"], p["output"])
        full = (o["plen"] + o["tlen"] <= MAXLEN) if (o["plen"] is not None and o["tlen"] is not None) else True
        rows.append((p["gid"], (gt, o["pt"], "" if gt == "<delete>" else gb, "" if o["pt"] == "<delete>" else o["text"], full)))
    return rows


TOP = {"keep": "keep", "edit": "edit", "delete": "delete", "rewrite": "rewrite", "<extract>": "keep", "<refine>": "edit",
       "<delete>": "delete", "<rewrite>": "rewrite"}
OPS = ["keep", "edit", "delete", "rewrite"]


def decision_flow(pages, outs):
    """teacher tag x student decision on every page; skipped / unparseable pages count as delete.
    -> (flow {teacher: {student: n}}, n_skipped, n_unparseable)"""
    flow = {t: {s: 0 for s in OPS} for t in OPS}
    for p, o in zip(pages, outs):
        flow[TOP[p["tag"]]][o["op"] if o["op"] in OPS else "delete"] += 1
    n_skip = sum(o["status"] != "ok" for o in outs); n_unp = sum(o["status"] == "ok" and o["op"] not in OPS for o in outs)
    return flow, n_skip, n_unp
