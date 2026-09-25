"""Stage 2a (CPU, DCLM env): RefinedWeb-rule on the resiliparse text of every page.

Stack = lib/rw_rule_chain.py (DCLM get_mapper(_safe=True) over dclm_baseline_refinedweb_post_lang.yaml, applied page by
page as DCLM's process_single_file does; an exception drops the page). Input = page["resiliparse"] (the pool
resiliparse_extract text, i.e. the input of the pool's RefinedWeb-rule stage) + WARC metadata. No lang/length gate is
applied: the pages come from the lang-filtered pool, and post_lang starts with page_length_filter.
On the first 960 pages the kept texts equal the pool's own RefinedWeb-rule output for the same WARC records (960/960
membership, 507/507 text).
Output <work>/sys_refinedweb_rule.jsonl {gid, text (null if dropped / no input), status, reason}.
usage: s2_rw_rule.py <page_table>
"""
import collections, json, os, sys
PT = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from common import *
W = workdir(PT)
from rw_rule_chain import steps_of, build, run      # (changes the working directory to $DCLM_DIR)

RW = build(steps_of("dclm_baseline_refinedweb_post_lang.yaml"))
print("RW steps:", [n for n, _ in RW], flush=True)
pages = load_pages(PT, ["gid", "stem", "url", "warc_id", "resiliparse"])
S = collections.Counter(); why = collections.Counter(); out = []
for p in pages:
    t = p["resiliparse"]
    o = {"gid": p["gid"], "text": None, "status": None, "reason": None}
    if not nonempty(t):
        o["status"] = "no_input"
    else:
        meta = {"WARC-Target-URI": p.get("url"), "WARC-Record-ID": p.get("warc_id")}
        res, reason = run(RW, {"text": t, "metadata": meta})
        txt = "\n".join(x["text"] for x in res) if res else ""
        o["text"] = txt if txt.strip() else None
        o["status"] = "kept" if o["text"] else "deleted"; o["reason"] = reason
        why[reason or "kept"] += 1
    S[o["status"]] += 1; out.append(o)
write_jsonl_atomic(W + "/sys_refinedweb_rule.jsonl", out)
V = {"counts": dict(S), "drop_reason": dict(why.most_common())}
json.dump(V, open(W + "/sys_refinedweb_rule.validation.json", "w"), indent=1)
print(json.dumps(V, indent=1)); print("RW_RULE_OK")
