"""Merged rule groups for Figure 1(a).

Reads rule_flags_5000.jsonl (one row per held-out page: keep-or-drop judge verdict, ReScraper's
operation, and the first failing rule of each rule stack; written by the rule-flag step of
evaluation/judges/, and released under analysis/keep_judge/ of the <HF_ORG>/<DATASET> dataset upon
acceptance). A page belongs to a
group when either stack drops it with a first-failing rule in that group, so pages both stacks
drop for the same kind of reason are counted once. For each group it prints the share of those
pages judged worth keeping (the rule bar) and the share that are worth keeping and that
ReScraper also drops (the ReScraper bar); both shares use the same denominator.

Usage: python analysis/analyze_rule_groups.py $WORK_DIR/eval/keep_judge/rule_flags_5000.jsonl
The printed n / rule shares are the values hard-coded in plot_rule_worth_keeping.py.
"""

import json
import sys

GROUPS = {
    "too many repetitions": (["massive_web_repetition_filters"],
                   ["gopher_rep_dup_line_frac", "fineweb_char_dup_ratio", "gopher_rep_duplicated_5_n_grams"]),
    "mostly numbers or symbols": (["alphabetic_word_ratio_filter"], ["gopher_qual_gopher_below_alpha_threshold"]),
    "page too short": (["page_length_filter"], ["gopher_qual_gopher_short_doc"]),
    "few lines end like sentences": ([], ["fineweb_line_punct_ratio"]),
    "too many noisy lines": (["word_removal_ratio_filter"], []),
}

rows = [json.loads(line) for line in open(sys.argv[1])]
for name, (rw, fw) in GROUPS.items():
    pages = [r for r in rows
             if str(r["refinedweb_rule_first_reason"]) in rw or str(r["fineweb_rule_first_reason"]) in fw]
    keep = [r for r in pages if r["judge_verdict"] == "keep"]
    both = [r for r in keep if str(r["rescraper_deletes"]) == "True"]
    print(f"{name:34s} n={len(pages):5d}  rule={100 * len(keep) / len(pages):6.2f}  "
          f"rescraper={100 * len(both) / len(pages):6.2f}")
