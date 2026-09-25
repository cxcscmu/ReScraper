"""Stage 2d helper. `keys`: write <work>/ux_keys.jsonl {gid, stem, md5, nchar} of the pages' resiliparse text.
`finish`: <work>/ux_out.jsonl (stages/s2_ultrax_join.py) -> <work>/sys_ultrax.jsonl {gid, text|null, status, n_same_text}
  text = UltraX `cleaned` (its post-process-and-execute output) when it has non-whitespace content; status:
  kept | deleted (UltraX emptied the page) | no_input (page has no resiliparse text) | not_found (text not in the run).
usage: s2_ultrax_finish.py keys|finish <page_table>"""
import collections, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from common import *
mode, PT = sys.argv[1], sys.argv[2]; W = workdir(PT)
pages = load_pages(PT, ["gid", "stem", "resiliparse"])
if mode == "keys":
    write_jsonl_atomic(W + "/ux_keys.jsonl", [{"gid": p["gid"], "stem": p["stem"], "md5": md5(p["resiliparse"]) if p["resiliparse"] is not None else None,
                                               "nchar": len(p["resiliparse"]) if p["resiliparse"] is not None else None} for p in pages])
    print("keys", len(pages), "stems", len({p["stem"] for p in pages})); sys.exit(0)
ux = load_by_gid(W + "/ux_out.jsonl"); assert len(ux) == len(pages)
S = collections.Counter(); out = []
for p in pages:
    u = ux[p["gid"]]
    if p["resiliparse"] is None or not p["resiliparse"].strip():
        o = {"gid": p["gid"], "text": None, "status": "no_input"}
    elif not u.get("found"):
        o = {"gid": p["gid"], "text": None, "status": "not_found", "why": u.get("why")}
    else:
        c = u.get("cleaned") or ""
        o = {"gid": p["gid"], "text": c if c.strip() else None, "status": "kept" if c.strip() else "deleted",
             "n_same_text": u.get("n_same_text"), "failed_program": "# FAILED" in (u.get("processed_functions") or "")}
    S[o["status"]] += 1; out.append(o)
write_jsonl_atomic(W + "/sys_ultrax.jsonl", out)
V = {"counts": dict(S), "dup_text_pages": sum(1 for p in pages if (ux[p["gid"]].get("n_same_text") or 1) > 1),
     "failed_program_pages": sum(bool(o.get("failed_program")) for o in out)}
json.dump(V, open(W + "/sys_ultrax.validation.json", "w"), indent=1)
print(json.dumps(V)); print("ULTRAX_OK")
