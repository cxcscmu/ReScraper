#!/usr/bin/env python3
"""Summarize the two-stage operations in a decision-first SFT JSONL file.

Table 6 (tab:sft-composition), Stage 2 columns: the `decision:<tag>` counts of the Stage-2 SFT set
(data_construction/sft_sets/build_stage2_set.py output). Rows are {input, output}; output line 1 is
the decision, then "<extract>", the extraction teacher's rm ops, the repeated decision and its payload.

usage: analyze_sft_operations.py <sft.jsonl> <out.json>
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


RM = re.compile(r"^rm (\d+)(?:-(\d+))?$")
SUB = re.compile(r'^sub (\d+): (".*")$')
DECISIONS = {"<keep>", "<edit>", "<delete>", "<rewrite>"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sft", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stats = Counter()
    edit_op_counts = []
    span_lengths = []

    with args.sft.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            lines = row["output"].splitlines()
            decision = lines[0]
            stats["rows"] += 1
            stats[f"decision:{decision}"] += 1
            try:
                boundary = next(i for i in range(2, len(lines)) if lines[i] in DECISIONS)
            except StopIteration:
                stats["malformed:no_second_decision"] += 1
                continue
            if lines[1] != "<extract>" or lines[boundary] != decision:
                stats["malformed:stage_or_repeat"] += 1
                continue
            extraction_ops = lines[2:boundary]
            final_payload = lines[boundary + 1 :]
            for op in extraction_ops:
                match = RM.match(op)
                if match:
                    stats["dripper_rm_ops"] += 1
                    stats["dripper_removed_lines"] += int(match.group(2) or match.group(1)) - int(match.group(1)) + 1
            if decision == "<edit>":
                count = 0
                has_rm = has_sub = False
                for op in final_payload:
                    match = RM.match(op)
                    if match:
                        has_rm = True
                        count += 1
                        stats["qwen_rm_ops"] += 1
                        stats["qwen_removed_lines"] += int(match.group(2) or match.group(1)) - int(match.group(1)) + 1
                        continue
                    match = SUB.match(op)
                    if match:
                        has_sub = True
                        count += 1
                        stats["qwen_sub_ops"] += 1
                        try:
                            span_lengths.append(len(json.loads(match.group(2))))
                        except json.JSONDecodeError:
                            stats["malformed:sub_json"] += 1
                edit_op_counts.append(count)
                stats[f"edit_mix:rm={has_rm},sub={has_sub}"] += 1

    def quantiles(values):
        values = sorted(values)
        if not values:
            return {}
        return {
            "mean": sum(values) / len(values),
            "p50": values[len(values) // 2],
            "p90": values[int(0.9 * (len(values) - 1))],
            "max": values[-1],
        }

    result = {
        "counts": dict(stats),
        "edit_operations_per_page": quantiles(edit_op_counts),
        "deleted_substring_characters": quantiles(span_lengths),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
