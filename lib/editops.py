"""Edit-op target format for text2text refinement (UltraX-style). Input lines are numbered "<lid:N> ...".
Target for <refine>: lines "rm A" / "rm A-B" (remove whole lines, 1-based) and "sub N: <removed substring>" (delete that
substring from line N; strict-subset contract). <extract>: tag only. <delete>: tag only. <rewrite>: tag + full text.
number_lines(text) / make_ops(src_text, dst_text) / apply_ops(numbered_or_plain_src, ops_text) -> body."""
import difflib, re, json
OPS_RM = re.compile(r"^rm (\d+)(?:-(\d+))?$"); OPS_SUB = re.compile(r"^sub (\d+): (.*)$"); LID = re.compile(r"^<lid:(\d+)> ?")
def lines_of(t): return [x.strip() for x in t.split("\n") if x.strip()]
def number_lines(t): return "\n".join(f"<lid:{i+1}> {x}" for i, x in enumerate(lines_of(t)))
def _removed_substrings(a, b):
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False); out = []
    for t, i1, i2, j1, j2 in sm.get_opcodes():
        if t == "delete": out.append(a[i1:i2])
        elif t in ("replace", "insert"): return None
    return out
def make_ops(src_text, dst_text):
    """Return (ops_text, exact) where exact=False means dst is not a strict subset edit of src (fallback to full text)."""
    src, dst = lines_of(src_text), lines_of(dst_text); sm = difflib.SequenceMatcher(a=src, b=dst, autojunk=False); ops = []
    for t, i1, i2, j1, j2 in sm.get_opcodes():
        if t == "equal": continue
        if t == "delete": ops.append(f"rm {i1+1}" if i2 - i1 == 1 else f"rm {i1+1}-{i2}"); continue
        if t == "replace" and (i2 - i1) == (j2 - j1):
            ok = True
            for k in range(i2 - i1):
                subs = _removed_substrings(src[i1 + k], dst[j1 + k])
                if subs is None or not subs or any(not s.strip() for s in subs): ok = False; break
                for s in subs: ops.append(f"sub {i1+k+1}: {json.dumps(s, ensure_ascii=False)}")
            if ok: continue
        return None, False
    return "\n".join(ops), True
def apply_ops(src_text, ops_text):
    src = [LID.sub("", x) for x in lines_of(src_text)]; rm = set(); subs = {}
    for l in ops_text.split("\n"):
        l = l.strip()
        if not l: continue
        m = OPS_RM.match(l)
        if m:
            a = int(m.group(1)); b = int(m.group(2) or a); rm.update(range(a, b + 1)); continue
        m = OPS_SUB.match(l)
        if m:
            try: s = json.loads(m.group(2))
            except Exception: s = m.group(2).strip("\"")
            subs.setdefault(int(m.group(1)), []).append(s)
    out = []
    for i, x in enumerate(src, 1):
        if i in rm: continue
        for s in subs.get(i, []): x = x.replace(s, "", 1)
        x = " ".join(x.split())
        if x: out.append(x)
    return "\n".join(out)
def split_output(o):
    o = o.lstrip(); tag = o.split("\n", 1)[0].strip(); body = o.split("\n", 1)[1] if "\n" in o else ""
    return tag, body
def to_editops_row(inp, out):
    tag, body = split_output(out); ninp = number_lines(inp)
    if tag == "<refine>":
        ops, exact = make_ops(inp, body)
        if exact and ops: return {"input": ninp, "output": "<edit>\n" + ops}, "refine_ops"
        if exact and not ops: return {"input": ninp, "output": "<keep>"}, "refine_noop_to_keep"
        return {"input": ninp, "output": "<edit>\n" + body}, "refine_fulltext_fallback"
    if tag == "<extract>": return {"input": ninp, "output": "<keep>"}, "keep"
    if tag == "<delete>": return {"input": ninp, "output": "<delete>"}, "delete"
    return {"input": ninp, "output": out}, tag.strip("<>")
def body_from_prediction(ninp, pred):
    """What the student output means as plain text (for scoring / corpus building)."""
    tag, body = split_output(pred)
    if tag == "<delete>": return tag, ""
    if tag in ("<keep>", "<extract>"): return "<extract>", apply_ops(ninp, "")
    if tag in ("<edit>", "<refine>"): return "<refine>", (apply_ops(ninp, body) if all(OPS_RM.match(l.strip()) or OPS_SUB.match(l.strip()) or not l.strip() for l in body.split("\n")) else body)
    return tag, body
