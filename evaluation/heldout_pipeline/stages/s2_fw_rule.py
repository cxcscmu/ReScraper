"""Stage 2b (CPU, datatrove 0.2.0 env): FineWeb-rule on the resiliparse text of every page.

Filter objects and arguments = lib/fw_rule_chain.py (the FineWeb-rule baseline's build()/res()). Per page:
  URLFilter(url)  ->  [extraction = page["resiliparse"], the same resiliparse text the baseline extracts itself]
  -> blank/whitespace-only lines removed ("\\n".join(l for l in t.splitlines() if l.strip()))  (the baseline's
     trafilatura-layout adapter)  -> LanguageFilter(en, 0.65, fastText lid.176.bin) -> GopherRepetitionFilter()
  -> GopherQualityFilter() -> C4QualityFilter(filter_no_terminal_punct=False) -> FineWebQualityFilter()
  (no MinHash dedup, no PII formatter - as in the baseline)
URLFilter.download_data() fetches datatrove's URL block lists on first use (needs network or a warm HF cache).
Output <work>/sys_fineweb_rule.jsonl {gid, text|null, status, reason, diag_raw_gopher_rep_drop}
usage: s2_fw_rule.py <page_table>
"""
import collections, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from common import *
from fw_rule_chain import build, res, Document
PT = sys.argv[1]; W = workdir(PT)
from importlib.metadata import version; print("datatrove", version("datatrove"), flush=True)
UF, FILTERS = build()
pages = load_pages(PT, ["gid", "url", "resiliparse"])
S = collections.Counter(); R = collections.Counter(); out = []
for p in pages:
    o = {"gid": p["gid"], "text": None, "status": None, "reason": None, "diag_raw_gopher_rep_drop": None}
    t = p["resiliparse"]; url = p.get("url")
    if url:
        ok, why = res(UF.filter(Document(text="", id=str(p["gid"]), metadata={"url": url})))
        if not ok:
            o.update(status="deleted", reason="url_" + str(why)); S["deleted"] += 1; R[o["reason"]] += 1; out.append(o); continue
    else:
        S["no_url"] += 1
    if not nonempty(t):
        o["status"] = "no_input"; S["no_input"] += 1; out.append(o); continue
    o["diag_raw_gopher_rep_drop"] = not res(FILTERS[1][1].filter(Document(text=t, id=str(p["gid"]), metadata={})))[0]
    t = "\n".join(l for l in t.splitlines() if l.strip())
    doc = Document(text=t, id=str(p["gid"]), metadata={})
    for name, f in FILTERS:
        try:
            ok, why = res(f.filter(doc))
        except Exception as e:
            ok, why = False, "error_" + type(e).__name__
        if not ok:
            o.update(status="deleted", reason="%s_%s" % (name, why)); break
    else:
        if doc.text.strip():
            o.update(status="kept", text=doc.text)
        else:
            o.update(status="deleted", reason="empty_after_c4")
    S[o["status"]] += 1; R[o["reason"] or "kept"] += 1; out.append(o)
write_jsonl_atomic(W + "/sys_fineweb_rule.jsonl", out)
V = {"counts": dict(S), "reasons": dict(R.most_common()), "diag_raw_gopher_rep_drop": sum(bool(o["diag_raw_gopher_rep_drop"]) for o in out),
     "kept_rate_of_pages_with_input": S["kept"] / max(1, len(pages) - S["no_input"])}
json.dump(V, open(W + "/sys_fineweb_rule.validation.json", "w"), indent=1)
print(json.dumps(V, indent=1)); print("FW_RULE_OK")
