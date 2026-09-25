"""Extraction-judge summary (Figure 8, right). usage: agg_extract.py <judge_out_dir> <N> [ext_texts.jsonl]
Reads <dir>/units/{extract,sanity}_*.jsonl. Writes <dir>/ext_judge_<N>.json (+ .perpage.jsonl, + sanity_50.json).
Per system: parse rate, mean of each 0-2 dimension, mean total (/6), share of pages with all three = 2, integrity over
non-empty outputs only; percentile bootstrap 95% CIs over pages (B=1000, RNG seeded per statistic); paired student-minus-system diffs.
Unparsed judgments are excluded from means (count reported). Sanity: expectation pass rates of the synthetic controls."""
import sys, os, json, glob, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from common import write_json_atomic, write_jsonl_atomic, is_empty  # noqa
D = sys.argv[1]; N = int(sys.argv[2]); ET = sys.argv[3] if len(sys.argv) > 3 else None
DIMS = ["main_content_recall", "boilerplate_precision", "integrity"]
SYSTEMS = ["student", "dripper", "resiliparse", "trafilatura", "justext"]
def load(pat):
    rows = []
    for p in sorted(glob.glob(D + "/units/" + pat)): rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    return rows
ex = load("extract_*.jsonl"); sa = load("sanity_*.jsonl")
empty = {}
if ET:
    for l in open(ET, encoding="utf-8"):
        t = json.loads(l)
        for s in SYSTEMS: empty[(t["gid"], s)] = is_empty(t[s])
import zlib
B = 1000
def ci(arr, key, fn=np.mean):
    """percentile bootstrap; the RNG is seeded per statistic (crc32 of key) so each CI depends only on its own data"""
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0: return None
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    bs = [fn(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return {"est": round(float(fn(arr)), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
by = collections.defaultdict(dict)
for r in ex: by[r["system"]][r["gid"]] = r
out = {"pages": N, "judge_dir": os.path.abspath(D), "requests": len(ex), "parsed": sum(r["parsed"] is not None for r in ex),
       "finish_length": sum(r["finish_reason"] == "length" for r in ex), "truncated_requests": sum(bool(r.get("truncated")) for r in ex),
       "model": ex[0]["model"] if ex else None, "gpu": sorted({r["gpu"] for r in ex}), "jobs": sorted({str(r["job"]) for r in ex}),
       "mean_out_tokens": round(float(np.mean([r["n_out_tokens"] for r in ex])), 1) if ex else None, "systems": {}}
out["parse_rate"] = round(out["parsed"] / max(1, out["requests"]), 5)
per = []
for s in SYSTEMS:
    rs = by.get(s, {}); ok = {g: r["parsed"] for g, r in rs.items() if r["parsed"] is not None}
    d = {"judged": len(rs), "parsed": len(ok)}
    for k in DIMS: d[k] = ci([v[k] for v in ok.values()], s + "/" + k)
    d["total_of_6"] = ci([sum(v[k] for k in DIMS) for v in ok.values()], s + "/total")
    d["all_three_2"] = ci([all(v[k] == 2 for k in DIMS) for v in ok.values()], s + "/all2")
    if empty:
        ne = [v["integrity"] for g, v in ok.items() if not empty.get((g, s), False)]
        d["integrity_nonempty_outputs"] = ci(ne, s + "/integrity_nonempty"); d["n_empty_outputs"] = sum(empty.get((g, s), False) for g in ok)
        d["recall_on_empty_outputs_mean"] = round(float(np.mean([v["main_content_recall"] for g, v in ok.items() if empty.get((g, s))])), 4) if d["n_empty_outputs"] else None
    out["systems"][s] = d
    for g, r in rs.items():
        per.append({"gid": g, "system": s, "pos": r.get("pos"), "scores": r["parsed"], "finish_reason": r["finish_reason"], "truncated": r.get("truncated")})
diffs = {}
st = {g: r["parsed"] for g, r in by.get("student", {}).items() if r["parsed"] is not None}
for s in SYSTEMS[1:]:
    ot = {g: r["parsed"] for g, r in by.get(s, {}).items() if r["parsed"] is not None}; gs = sorted(set(st) & set(ot))
    for k in DIMS + ["total"]:
        f = (lambda v: sum(v[x] for x in DIMS)) if k == "total" else (lambda v, k=k: v[k])
        diffs["student-%s_%s" % (s, k)] = ci([f(st[g]) - f(ot[g]) for g in gs], "diff/%s/%s" % (s, k))
out["paired_diffs_student_minus"] = diffs
pos = collections.defaultdict(list)
for r in ex:
    if r["parsed"] is not None: pos[r["pos"]].append(sum(r["parsed"][k] for k in DIMS))
out["mean_total_by_submission_position"] = {str(p): round(float(np.mean(v)), 3) for p, v in sorted(pos.items())}
# ---- sanity controls
EXPECT = {"ctl_empty": ("main_content_recall == 0", lambda v: v["main_content_recall"] == 0),
          "ctl_full": ("boilerplate_precision <= 1", lambda v: v["boilerplate_precision"] <= 1),
          "ctl_first40": ("main_content_recall <= 1", lambda v: v["main_content_recall"] <= 1),
          "ctl_twice": ("integrity <= 1", lambda v: v["integrity"] <= 1),
          "ctl_shuffled": ("integrity <= 1", lambda v: v["integrity"] <= 1)}
san = {"requests": len(sa), "parsed": sum(r["parsed"] is not None for r in sa), "controls": {}}
fails = []
for c, (desc, fn) in EXPECT.items():
    rs = [r for r in sa if r["system"] == c and r["parsed"] is not None]
    passed = [fn(r["parsed"]) for r in rs]
    san["controls"][c] = {"expectation": desc, "n": len(rs), "pass": int(sum(passed)), "pass_rate": round(sum(passed) / max(1, len(rs)), 4),
                          "mean_scores": {k: round(float(np.mean([r["parsed"][k] for r in rs])), 3) for k in DIMS} if rs else None}
    fails += [{"gid": r["gid"], "control": c, "scores": {k: r["parsed"][k] for k in DIMS}, "rationale": r["parsed"]["rationale"]} for r, p in zip(rs, passed) if not p]
sg = sorted({r["gid"] for r in sa})
san["pages"] = sg
san["real_systems_on_sanity_pages"] = {s: {k: round(float(np.mean([by[s][g]["parsed"][k] for g in sg if g in by.get(s, {}) and by[s][g]["parsed"]])), 3) for k in DIMS}
                                      for s in SYSTEMS if by.get(s)}
san["failures"] = fails
out["sanity"] = {k: v for k, v in san.items() if k != "failures"}; out["sanity"]["n_failures"] = len(fails)
write_json_atomic("%s/ext_judge_%d.json" % (D, N), out); write_jsonl_atomic("%s/ext_judge_%d.perpage.jsonl" % (D, N), per)
write_json_atomic("%s/sanity_50.json" % D, san)
f = lambda x: "%.3f [%.3f, %.3f]" % (x["est"], x["lo"], x["hi"]) if x else "-"
print("requests %d parsed %d (%.2f%%) finish=length %d truncated %d" % (out["requests"], out["parsed"], 100 * out["parse_rate"], out["finish_length"], out["truncated_requests"]))
for s, d in out["systems"].items():
    print("%-12s recall %s  precision %s  integrity %s  total %s  all2 %s" % (s, f(d["main_content_recall"]), f(d["boilerplate_precision"]), f(d["integrity"]), f(d["total_of_6"]), f(d["all_three_2"])))
print("sanity:", {c: "%d/%d" % (v["pass"], v["n"]) for c, v in san["controls"].items()}, "parsed %d/%d" % (san["parsed"], san["requests"]))
print("AGG_EXTRACT_DONE")
