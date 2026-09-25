"""FineWeb-rule on the 5,000 held-out pages, every sub-rule evaluated (no early stop).

Same code, filter objects, arguments and inputs as heldout_pipeline/stages/s2_fw_rule.py (lib/fw_rule_chain.py build():
URLFilter -> blank lines removed -> lang -> gopher_rep -> gopher_qual -> c4 -> fineweb, datatrove 0.2.0). Each datatrove
filter returns at its first failing check; here the filter() source of gopher_rep / gopher_qual / c4 / fineweb is
rewritten mechanically so that every `return False, <reason>` appends the reason and continues, and
`return True` returns the list. So every check is evaluated on the text the real stack would hand it had no
earlier check fired. c4 still rewrites the text (it also keeps lines that contain "{" / "lorem ipsum" instead of
dropping the page), and fineweb runs on the c4-rewritten text, as in the stack.
Flag names = the reason strings of s2_fw_rule.py ("<filter>_<reason>").
Validation: the first fired flag in stack order equals the recorded reason in <work>/sys_fineweb_rule.jsonl.
Interpreter: datatrove 0.2.0 env (DATATROVE_PY). Output: <out_dir>/fw_flags.jsonl {gid, fw_fired, fw_incomplete}.
usage: fw_allrules.py <heldout5k.jsonl> <heldout_pipeline work dir> <out_dir>
"""
import collections, inspect, json, os, re, sys, textwrap, types
PT, WK, OUT = (os.path.abspath(x) for x in sys.argv[1:4])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "heldout_pipeline", "lib"))
from common import load_pages, read_jsonl, write_jsonl_atomic, nonempty
from fw_rule_chain import build, res, Document
REF = WK + "/sys_fineweb_rule.jsonl"
UF, FILTERS = build()


def all_checks(obj):
    """Bind a copy of obj.filter that collects every failing check instead of returning at the first."""
    src = textwrap.dedent(inspect.getsource(type(obj).filter))
    n_ret = len(re.findall(r"return False, ", src))
    src2 = re.sub(r"return False, (.+?)(\s*#.*)?$", r"_H.append(\1)", src, flags=re.M)
    src2 = re.sub(r"^(\s*)return True\s*$", r"\1return _H", src2, flags=re.M)
    lines = src2.split("\n"); d = next(i for i, l in enumerate(lines) if l.rstrip().endswith(":") and l.startswith("def "))
    body_indent = re.match(r"\s*", lines[d + 1]).group(0)
    lines.insert(d + 1, body_indent + "_H = self._H = []")
    src2 = "\n".join(lines)
    assert "return False" not in src2 and src2.count("_H.append(") == n_ret, src2
    g = sys.modules[type(obj).__module__].__dict__
    ns = {}; exec(src2, g, ns)
    return types.MethodType(ns["filter"], obj)


CHAIN = []
for name, f in FILTERS:
    CHAIN.append((name, f, None if name == "lang" else all_checks(f)))
print("FW chain:", [n for n, _, _ in CHAIN], flush=True)

pages = load_pages(PT, ["gid", "url", "resiliparse"])
ref = {r["gid"]: r for r in read_jsonl(REF)}
out, bad, cnt = [], [], collections.Counter()
for p in pages:
    o = {"gid": p["gid"], "fw_no_input": False, "fw_fired": [], "fw_incomplete": None}
    url, t = p.get("url"), p["resiliparse"]
    if url:
        ok, why = res(UF.filter(Document(text="", id=str(p["gid"]), metadata={"url": url})))
        if not ok: o["fw_fired"].append("url_" + str(why))
    if not nonempty(t):
        o["fw_no_input"] = True
    else:
        t = "\n".join(l for l in t.splitlines() if l.strip())
        doc = Document(text=t, id=str(p["gid"]), metadata={})
        for name, f, allf in CHAIN:
            if allf is None:
                ok, why = res(f.filter(doc))
                if not ok: o["fw_fired"].append("%s_%s" % (name, why))
                continue
            try:
                hits = allf(doc)
            except Exception as e:          # a later check divides by zero on an (almost) empty page
                hits = list(f._H)            # checks up to the failing line; the rest cannot be evaluated
                o["fw_incomplete"] = "%s:%s" % (name, type(e).__name__)
                if not hits: hits = ["error_" + type(e).__name__]
            o["fw_fired"] += ["%s_%s" % (name, h) for h in dict.fromkeys(hits)]
        if not o["fw_fired"] and not doc.text.strip():
            o["fw_fired"].append("empty_after_c4")
    r = ref[p["gid"]]
    first = o["fw_fired"][0] if o["fw_fired"] else None
    if r["status"] == "no_input": ok = o["fw_no_input"] and not o["fw_fired"]
    else: ok = first == r["reason"] and ((r["status"] == "deleted") == bool(o["fw_fired"]))
    if not ok: bad.append((p["gid"], r["status"], r["reason"], o["fw_fired"]))
    cnt.update(o["fw_fired"]); out.append(o)
os.makedirs(OUT, exist_ok=True)
write_jsonl_atomic(OUT + "/fw_flags.jsonl", out)
print("fired counts (independent):", dict(cnt.most_common()))
print("incomplete:", sum(1 for o in out if o["fw_incomplete"]), collections.Counter(o["fw_incomplete"] for o in out if o["fw_incomplete"]))
print("mismatch vs recorded first reason:", len(bad), bad[:10])
print("FW_ALL_OK" if not bad else "FW_ALL_MISMATCH", flush=True)
