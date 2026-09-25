"""Execution and operation classification of one ReScraper program on its page (shared by the greedy run,
stages/s1_ours_exec.py, and the release-decoding run, stages/s1_rel_exec.py).

Per page (lib/e2e_ops.py + lib/editops.py, the executor of the pool writer):
  decision  = line 1 of the raw program (<keep>/<edit>/<delete>/<rewrite>; anything else = unparseable)
  pt, body  = e2e_ops.body_from_prediction_dfirst(input, raw)       (the decision-first reader)
  text      = "" if pt == <delete> else body
  extracted = editops.apply_ops(input, ops1), ops1 = parse_staged(lines[1:])[0]   (pre-op extraction; the "before"
              text of the operation-score groups and of the operation mix)
  opmix_cls = classify_ours() below
"""
from common import e2e, OPNAME

X, E = e2e()
norm = lambda t: " ".join((t or "").split())


def classify_ours(ninp, pred):
    """operation-mix class of one ReScraper program (status handled by the caller) -> (class, detail, extracted, text, emits)"""
    lines = pred.strip().split("\n"); d = lines[0].strip() if lines else ""
    tag, text = X.body_from_prediction_dfirst(ninp, pred)
    emits = not (tag == "<delete>" or not (text or "").strip())
    det = {}
    if d not in X.DFIRST_TAGS:
        det.update(reason="unparseable (no decision on line 1)", staged_fallback_tag=tag, pool_writer_would_emit=emits)
        return "omitted", det, None, text if emits else "", emits
    ops1, tag2, rest = X.parse_staged("\n".join(lines[1:]))
    extracted = E.apply_ops(ninp, ops1)
    det["stage1_rm"] = sum(bool(E.OPS_RM.match(l.strip())) for l in ops1.split("\n"))
    det["stage1_sub"] = sum(bool(E.OPS_SUB.match(l.strip())) for l in ops1.split("\n"))
    det["tail_tag"] = tag2 if tag2 in X.DFIRST_TAGS else ("<none>" if not tag2 else "<non-tag>")
    if d == "<delete>": return "delete", det, extracted, "", False
    if d == "<keep>": return "keep", det, extracted, text, emits
    if tag2 not in X.DFIRST_TAGS: rest = (tag2 + "\n" + rest) if tag2 else rest
    if d == "<rewrite>":
        if not emits: det["note"] = "rewrite with empty payload (pool writer drops the page)"
        return "rewrite", det, extracted, text, emits
    bl = [l.strip() for l in rest.split("\n") if l.strip()]
    if not bl:
        det["edit_kind"] = "empty payload"; c = "edit: empty payload"
    else:
        ok = [l for l in bl if E.OPS_RM.match(l) or E.OPS_SUB.match(l)]
        if len(ok) == len(bl):
            n_rm = sum(bool(E.OPS_RM.match(l)) for l in ok); n_sub = len(ok) - n_rm
            det.update(stage2_rm=n_rm, stage2_sub=n_sub)
            c = "edit: lines only" if n_sub == 0 else "edit: strings only" if n_rm == 0 else "edit: lines + strings"
        else:
            det.update(stage2_lines=len(bl), stage2_op_lines=len(ok))
            c = "edit: full-text fallback"
    if c != "edit: empty payload" and norm(text) == norm(extracted):
        det["edit_kind"] = c; c = "edit, no effective change"
    return c, det, extracted, text, emits


def execute(page, r):
    o = {"gid": page["gid"], "status": r["status"], "decision": None, "pt": None, "op": None, "text": None,
         "extracted": None, "opmix_cls": None, "opmix_det": None, "emits": False,
         "gen_tokens": r.get("gen_tokens"), "finish_reason": r.get("finish_reason"), "plen": r.get("plen"), "tlen": r.get("tlen")}
    if r["status"] != "ok":
        o["opmix_cls"] = "omitted"; o["opmix_det"] = {"reason": r["status"]}
        return o
    raw = r["raw"]
    lines = raw.strip().split("\n"); d = lines[0].strip() if lines else ""
    pt, body = X.body_from_prediction_dfirst(page["input"], raw)
    c, det, extracted, text, emits = classify_ours(page["input"], raw)
    o.update(decision=d, pt=pt, op=OPNAME.get(d) if d in X.DFIRST_TAGS else None,
             text=("" if pt == "<delete>" else body), extracted=extracted, opmix_cls=c, opmix_det=det, emits=emits)
    return o
