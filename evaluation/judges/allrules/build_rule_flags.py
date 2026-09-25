"""Per-page table for Figure 1(a) (rule_flags_5000.jsonl): judge verdict, every rule each stack fires, and what ReScraper
does. Inputs: <out_dir>/rw_flags.jsonl and fw_flags.jsonl (rw_allrules.py / fw_allrules.py), keep_judge_5000.jsonl
(../keep_judge.py), <work>/{sys_refinedweb_rule,sys_fineweb_rule,ours}.jsonl of the release-decoding held-out run.
Output: <out_dir>/rule_flags_5000.jsonl (columns documented in README.md), read by analysis/analyze_rule_groups.py.
usage: build_rule_flags.py <heldout_pipeline work dir> <keep_judge_5000.jsonl> <out_dir>"""
import collections, json, sys
W, KJ, D = sys.argv[1].rstrip("/") + "/", sys.argv[2], sys.argv[3].rstrip("/") + "/"
rd = lambda p: {int(r["gid"]): r for r in map(json.loads, open(p))}
RWF, FWF = rd(D + "rw_flags.jsonl"), rd(D + "fw_flags.jsonl")
RWS, FWS, O = rd(W + "sys_refinedweb_rule.jsonl"), rd(W + "sys_fineweb_rule.jsonl"), rd(W + "ours.jsonl")
K = rd(KJ)
# the rules shown in Figure 1 (>= 80 first-fired drops); column = stack prefix + reason string of the stack
FIG1 = {"rw_massive_web_repetition_filters": "repeated lines / n-grams",
        "rw_word_removal_ratio_filter": "cleaning cut > 5% -> page dropped",
        "rw_page_length_filter": "page too short",
        "rw_alphabetic_word_ratio_filter": "few alphabetic words",
        "fw_gopher_qual_gopher_below_alpha_threshold": "< 80% of words have a letter",
        "fw_fineweb_char_dup_ratio": "duplicated characters",
        "fw_gopher_rep_dup_line_frac": "duplicate lines",
        "fw_fineweb_line_punct_ratio": "few lines end in punctuation",
        "fw_gopher_rep_duplicated_5_n_grams": "duplicated 5-grams",
        "fw_lang_None": "not detected as English",
        "fw_gopher_qual_gopher_short_doc": "document too short"}
kept = lambda r: bool(r.get("text") and r["text"] != "None" and r["text"].strip())
rows, c = [], collections.Counter()
for g in sorted(K):
    o = O[g]; ok = o.get("status") == "ok"
    r = {"gid": g, "judge_verdict": K[g]["parsed"]["verdict"],
         "rescraper_op": o["op"] if ok else o["status"],
         "rescraper_deletes": not (ok and o["op"] != "delete"),
         "refinedweb_rule_kept": kept(RWS[g]), "fineweb_rule_kept": kept(FWS[g]),
         "refinedweb_rule_first_reason": RWS[g]["reason"], "fineweb_rule_first_reason": FWS[g]["reason"]}
    rw = set("rw_" + x for x in RWF[g]["rw_fired"]); fw = set("fw_" + x for x in FWF[g]["fw_fired"])
    for k in FIG1: r[k] = k in rw or k in fw
    r["refinedweb_rule_all_fired"] = RWF[g]["rw_fired"]; r["fineweb_rule_all_fired"] = FWF[g]["fw_fired"]
    r["fineweb_rule_eval_incomplete"] = FWF[g]["fw_incomplete"]
    assert r["refinedweb_rule_kept"] == (not RWF[g]["rw_fired"] and not RWF[g]["rw_no_input"]), g
    assert r["fineweb_rule_kept"] == (not FWF[g]["fw_fired"] and not FWF[g]["fw_no_input"]), g
    rows.append(r)
    for k in FIG1: c[k, "fired"] += r[k]; c[k, "first"] += r[k] and (r[("refinedweb" if k[:2] == "rw" else "fineweb") + "_rule_first_reason"] == k[3:])
with open(D + "rule_flags_5000.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("n", len(rows), "judge keep", sum(r["judge_verdict"] == "keep" for r in rows), "rescraper deletes", sum(r["rescraper_deletes"] for r in rows))
for k, lab in FIG1.items(): print(f"{k:48s} fired {c[k,'fired']:5d}  first-fired {c[k,'first']:5d}  ({lab})")
