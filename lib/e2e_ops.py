"""Staged edit-op format for the end-to-end text cleaner: the model sees the numbered full-page text and
emits (1) an <extract> block = dripper line removals, then (2) the 27B/rewrite decision on the same
line ids: <keep> | <edit> + ops | <delete> | <rewrite> + text. Both stages use the original ids, so
the reader just applies the union of the ops. Each stage can be scored on its own."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import editops as E

# An <edit> body is applied as ops only when EVERY line parses as one; otherwise the block is taken
# as the page text. So a generation that runs out of tokens in the middle of an op line writes the op
# list itself into the corpus - measured on 25 shipped shards: 38 pages of 44,741 (0.085%) and 0.194%
# of words are literal sub 36: "..." lines. With this on, the ops that did parse are applied and the
# truncated tail is dropped. Off by default so a queued job's output stays bit-identical; set
# OPS_TOLERANT=1 for new inference runs.
OPS_TOLERANT = os.environ.get("OPS_TOLERANT", "") == "1"
OPS_TOLERANT_MIN = float(os.environ.get("OPS_TOLERANT_MIN", "0.8"))


def _apply_edit_body(ninp, ops1, rest):
    """(text, True) if the block was read as edit ops, (raw text, False) if it is page text."""
    bl = [l.strip() for l in rest.split("\n") if l.strip()]
    if not bl:
        return rest.strip(), False
    ok = [l for l in bl if E.OPS_RM.match(l) or E.OPS_SUB.match(l)]
    if len(ok) == len(bl) or (OPS_TOLERANT and ok and len(ok) >= OPS_TOLERANT_MIN * len(bl)):
        return E.apply_ops(ninp, ops1 + "\n" + "\n".join(ok)), True
    return rest.strip(), False

def _ranges(ids):
    out = []; ids = sorted(ids)
    while ids:
        a = b = ids.pop(0)
        while ids and ids[0] == b + 1: b = ids.pop(0)
        out.append(f"rm {a}" if a == b else f"rm {a}-{b}")
    return out

def to_staged_row(full, drip, out):
    """full = webkit_txt(raw page), drip = dripper text, out = teacher target (<extract>/<refine>/<delete>/<rewrite> + body)."""
    ninp = E.number_lines(full)
    ops1, exact1 = E.make_ops(full, drip)
    if not exact1: return None, "stage1_not_subset"
    rm = set()
    for l in ops1.split("\n"):
        l = l.strip()
        if not l: continue
        m = E.OPS_RM.match(l)
        if not m: return None, "stage1_has_sub"
        a = int(m.group(1)); b = int(m.group(2) or a); rm.update(range(a, b + 1))
    n_full = len(E.lines_of(full)); keep_ids = [i for i in range(1, n_full + 1) if i not in rm]
    if len(keep_ids) != len(E.lines_of(drip)): return None, "stage1_len_mismatch"
    head = "<extract>" + ("\n" + ops1 if ops1 else "")
    tag, body = E.split_output(out)
    if tag == "<delete>": return {"input": ninp, "output": head + "\n<delete>"}, "delete"
    if tag == "<rewrite>": return {"input": ninp, "output": head + "\n<rewrite>\n" + body.strip()}, "rewrite"
    if tag == "<extract>": return {"input": ninp, "output": head + "\n<keep>"}, "keep"
    ops2, exact2 = E.make_ops(drip, body)
    if not exact2: return {"input": ninp, "output": head + "\n<edit>\n" + body.strip()}, "edit_fulltext_fallback"
    if not ops2: return {"input": ninp, "output": head + "\n<keep>"}, "refine_noop_to_keep"
    tr = []
    for l in ops2.split("\n"):
        l = l.strip()
        if not l: continue
        m = E.OPS_RM.match(l)
        if m:
            a = int(m.group(1)); b = int(m.group(2) or a); tr += _ranges([keep_ids[k - 1] for k in range(a, b + 1)]); continue
        m = E.OPS_SUB.match(l); tr.append(f"sub {keep_ids[int(m.group(1)) - 1]}: {m.group(2)}")
    return {"input": ninp, "output": head + "\n<edit>\n" + "\n".join(tr)}, "edit_ops"

def parse_staged(pred):
    """-> (stage1_ops_text, final_tag, final_body)"""
    lines = pred.strip().split("\n"); i = 0; ops = []
    if lines and lines[0].strip() == "<extract>":
        i = 1
        while i < len(lines):
            s = lines[i].strip()
            if not s or E.OPS_RM.match(s) or E.OPS_SUB.match(s): ops.append(s); i += 1
            else: break
    if i >= len(lines): return "\n".join(ops), "<keep>", ""
    return "\n".join(ops), lines[i].strip(), "\n".join(lines[i + 1:])

def body_from_prediction_staged(ninp, pred):
    """What the prediction means as text: (teacher-style tag, text). Stage-1 ops are applied for keep/edit."""
    ops1, tag, rest = parse_staged(pred)
    if tag == "<delete>": return "<delete>", ""
    if tag == "<rewrite>": return "<rewrite>", rest.strip()
    if tag == "<keep>": return "<extract>", E.apply_ops(ninp, ops1)
    if tag == "<edit>":
        return "<refine>", _apply_edit_body(ninp, ops1, rest)[0]
    return tag, rest.strip()

def stage1_removed_ids(ops_text):
    rm = set()
    for l in ops_text.split("\n"):
        m = E.OPS_RM.match(l.strip())
        if m: a = int(m.group(1)); b = int(m.group(2) or a); rm.update(range(a, b + 1))
    return rm

if __name__ == "__main__":
    full = "Home Login Register\nBest Widgets 2011\nThe widget is fast [Share on Twitter] and red.\nIt costs 5 dollars.\nFollow us\nCopyright 2011 Widgets Inc"
    drip = "Best Widgets 2011\nThe widget is fast [Share on Twitter] and red.\nIt costs 5 dollars."
    cases = [("<extract>\n" + drip, "keep"),
             ("<refine>\nBest Widgets 2011\nThe widget is fast and red.\nIt costs 5 dollars.", "edit_ops"),
             ("<refine>\nBest Widgets 2011\nIt costs 5 dollars.", "edit_ops"),
             ("<refine>\nBest Widgets 2011\nThe widget is fast and red.", "edit_fulltext_fallback"),
             ("<delete>", "delete"), ("<rewrite>\nWidgets are fast and red and cost five dollars.", "rewrite")]
    for out, want in cases:
        row, kind = to_staged_row(full, drip, out); assert kind == want, (kind, want)
        tag, text = body_from_prediction_staged(row["input"], row["output"])
        target = E.split_output(out)[1].strip() if want != "keep" else drip
        if want == "delete": target = ""
        assert " ".join(text.split()) == " ".join(target.split()), (want, text, target)
        print(want, "OK ->", row["output"].replace("\n", " | "))
    print("SELFTEST_OK")


# --- decision-first (ALIGN_DFIRST=1) ------------------------------------------------
DFIRST_TAGS = {"<keep>": "<extract>", "<edit>": "<refine>", "<delete>": "<delete>", "<rewrite>": "<rewrite>"}

def body_from_prediction_dfirst(ninp, pred):
    """Decision-first prediction -> (teacher-style tag, text).

    Line 1 is the decision and is what we score; the rest is an ordinary staged target and
    supplies the body. If line 1 is not a bare tag the model did not learn the format, so we
    fall back to the staged reading rather than silently scoring a wrong tag."""
    lines = pred.strip().split("\n")
    d = lines[0].strip() if lines else ""
    if d not in DFIRST_TAGS:
        return body_from_prediction_staged(ninp, pred)
    ops1, tag2, rest = parse_staged("\n".join(lines[1:]))
    if d == "<delete>":
        return "<delete>", ""
    if d == "<keep>":
        return "<extract>", E.apply_ops(ninp, ops1)
    # the trailing tag should repeat the decision; when it does not, treat everything the
    # stage-1 block did not consume as the payload so a dropped repeat costs no body score.
    if tag2 not in DFIRST_TAGS:
        rest = (tag2 + "\n" + rest) if tag2 else rest
    if d == "<rewrite>":
        return "<rewrite>", rest.strip()
    return "<refine>", _apply_edit_body(ninp, ops1, rest)[0]


# --- merged 3-way decision head (ALIGN_DFIRST3=1) -----------------------------------
DFIRST3_LEAD = {"<clean>", "<delete>", "<rewrite>"}

def body_from_prediction_dfirst3(ninp, pred):
    """3-way-head prediction -> (teacher-style tag, text).

    Line 1 is one of <clean>/<delete>/<rewrite>. <clean> defers the extract-vs-refine call
    to the in-slot tag, so we simply drop line 1 and read the remainder as an ordinary
    staged target. <delete>/<rewrite> are decided on line 1 exactly as in ALIGN_DFIRST.
    A leading <keep>/<edit> (off-format but unambiguous) is honoured via the dfirst reader.
    Anything else means the model did not learn the format -> staged fallback, never a
    silently wrong tag."""
    lines = pred.strip().split("\n")
    d = lines[0].strip() if lines else ""
    if d == "<clean>":
        return body_from_prediction_staged(ninp, "\n".join(lines[1:]))
    if d in ("<delete>", "<rewrite>", "<keep>", "<edit>"):
        return body_from_prediction_dfirst(ninp, pred)
    return body_from_prediction_staged(ninp, pred)


# --- materialised stage-1 text in the target (ALIGN_ECHO=1) -------------------------
ECHO_DEC = ("<keep>", "<edit>", "<delete>", "<rewrite>")

def parse_echo(pred):
    """-> (stage1_ops_text, echoed_text_or_None, final_tag, final_body)"""
    lines = pred.strip().split("\n")
    i = 0; ops = []
    if lines and lines[0].strip() == "<extract>":
        i = 1
        while i < len(lines):
            s = lines[i].strip()
            if not s or E.OPS_RM.match(s) or E.OPS_SUB.match(s): ops.append(s); i += 1
            else: break
    echo = None
    if i < len(lines) and lines[i].strip() == "<text>":
        i += 1; body = []
        while i < len(lines) and lines[i].strip() not in ECHO_DEC:
            body.append(lines[i]); i += 1
        echo = "\n".join(body)
    if i >= len(lines):
        return "\n".join(ops), echo, "<keep>", ""
    return "\n".join(ops), echo, lines[i].strip(), "\n".join(lines[i + 1:])

def body_from_prediction_echo(ninp, pred):
    """Echo-format prediction -> (teacher-style tag, text).

    The <text> block is scaffolding, not output: it is dropped and the rest is read by the
    ordinary staged reader, so tag and body are scored by the same code as every other C run.
    A prediction with no <text> block degrades to the plain staged reading."""
    ops1, echo, tag, rest = parse_echo(pred)
    head = "<extract>" + ("\n" + ops1 if ops1 else "")
    return body_from_prediction_staged(ninp, head + "\n" + tag + ("\n" + rest if rest else ""))
