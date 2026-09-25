"""RefinedWeb-rule on the 5,000 held-out pages, every rule evaluated (no early stop).

Same mappers, arguments and inputs as heldout_pipeline/stages/s2_rw_rule.py (lib/rw_rule_chain.py: DCLM
get_mapper(_safe=True) from dclm_baseline_refinedweb_post_lang.yaml, page = resiliparse text + WARC metadata).
Difference: when a filter drops the page we record it and continue with the page as it was before that filter, so every
filter is evaluated on the same text the real stack would have given it had no earlier filter fired. Line modifiers
are applied as in the stack (they run after all filters and before word_removal_ratio_filter). If a modifier empties
the page, it is recorded as fired (and word_removal_ratio_filter as fired: 100% of words removed).
Validation: the first fired step in stack order equals the recorded reason in <work>/sys_refinedweb_rule.jsonl.
Interpreter: DCLM env (DCLM_PY). Output: <out_dir>/rw_flags.jsonl {gid, rw_fired: [step names in stack order]}.
usage: rw_allrules.py <heldout5k.jsonl> <heldout_pipeline work dir> <out_dir>
"""
import collections, json, os, sys
PT, WK, OUT = (os.path.abspath(x) for x in sys.argv[1:4])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "heldout_pipeline", "lib"))
from common import load_pages, read_jsonl, write_jsonl_atomic, nonempty
from rw_rule_chain import steps_of, build      # (changes the working directory to $DCLM_DIR)
REF = WK + "/sys_refinedweb_rule.jsonl"
RW = build(steps_of("dclm_baseline_refinedweb_post_lang.yaml"))
NAMES = [n for n, _ in RW]
print("RW steps:", NAMES, flush=True)
MODS = {n for n in NAMES if n.endswith("_modifier") or n.endswith("_enricher")}


def run_all(page):
    fired = []
    cur = [page]
    for name, f in RW:
        nxt, err = [], None
        for p in cur:
            r = f(dict(p, metadata=dict(p["metadata"])))
            if isinstance(r, list): nxt.extend(r)
            else: err = str(r)[:200]
        if err is not None:
            fired.append(name); continue          # _safe error = the stack drops the page
        if not nxt:
            fired.append(name)
            if name in MODS:                       # page emptied by cleaning: nothing left to test
                fired.append("word_removal_ratio_filter"); break
            continue                               # filter: keep testing the unfiltered page
        cur = nxt
    return list(dict.fromkeys(fired))


pages = load_pages(PT, ["gid", "url", "warc_id", "resiliparse"])
ref = {r["gid"]: r for r in read_jsonl(REF)}
out, bad, cnt = [], [], collections.Counter()
for p in pages:
    o = {"gid": p["gid"], "rw_no_input": not nonempty(p["resiliparse"]), "rw_fired": []}
    if not o["rw_no_input"]:
        meta = {"WARC-Target-URI": p.get("url"), "WARC-Record-ID": p.get("warc_id")}
        o["rw_fired"] = run_all({"text": p["resiliparse"], "metadata": meta})
    r = ref[p["gid"]]
    first = o["rw_fired"][0] if o["rw_fired"] else None
    rr = r["reason"].split(" ERROR ")[0] if r["reason"] else None
    if r["status"] == "no_input": ok = o["rw_no_input"]
    else: ok = (first == rr) and ((r["status"] == "deleted") == bool(o["rw_fired"]))
    if not ok: bad.append((p["gid"], r["status"], r["reason"], o["rw_fired"]))
    cnt.update(o["rw_fired"]); out.append(o)
os.makedirs(OUT, exist_ok=True)
write_jsonl_atomic(OUT + "/rw_flags.jsonl", out)
print("fired counts (independent):", dict(cnt.most_common()))
print("mismatch vs recorded first reason:", len(bad), bad[:10])
print("RW_ALL_OK" if not bad else "RW_ALL_MISMATCH", flush=True)
