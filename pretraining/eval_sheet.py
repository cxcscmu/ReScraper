import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd


low_variance_datasets = [
    "hellaswag_zeroshot",
    "jeopardy",
    "bigbench_qa_wikidata",
    "arc_easy",
    "arc_challenge",
    "copa",
    "commonsense_qa",
    "piqa",
    "openbook_qa",
    "lambada_openai",
    "hellaswag",
    "winograd",
    "winogrande",
    "bigbench_dyck_languages",
    "agi_eval_lsat_ar",
    "bigbench_cs_algorithms",
    "bigbench_operators",
    "bigbench_repeat_copy_logic",
    "squad",
    "coqa",
    "boolq",
    "bigbench_language_identification",
    "mmlu_fewshot",
]


def gen_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate eval JSONs into one tab-separated file for Google Sheets."
    )
    parser.add_argument(
        "--eval_meta_data",
        default="eval/eval_meta_data.csv",
        help="Eval meta data file",
    )
    parser.add_argument(
        "--eval_results",
        nargs="+",
        required=True,
        help="One or more eval result JSON files (same schema as metrics_10pct.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Single TSV output path (default: eval_results/google_sheet.txt next to this script)",
    )
    return parser


def compute_row(data, eval_metadata_df):
    """Return (row_label, ordered column names, list of cell values aligned to columns)."""
    eval_metadata = eval_metadata_df.copy()
    eval_metadata["results"] = eval_metadata["Eval Task"].map(
        data["eval_metrics"]["icl"]
    )
    eval_metadata["centered results"] = (
        eval_metadata["results"].astype(float)
        - 0.01 * eval_metadata["Random baseline"].astype(float)
    ) / (1.0 - 0.01 * eval_metadata["Random baseline"].astype(float))
    eval_metadata = eval_metadata[eval_metadata["Eval Task"] != "commonsense_qa"]

    task_categories = sorted(eval_metadata["Task Category"].unique())
    task_categories.append("Core")

    values_by_cat: dict = {}
    for c in task_categories:
        if c == "Core":
            eval_df = eval_metadata
        else:
            eval_df = eval_metadata[eval_metadata["Task Category"] == c]
        sorted_df = eval_df.sort_values(by="Eval Task")
        filtered_df = sorted_df[sorted_df["Eval Task"].isin(low_variance_datasets)]
        c_sum, c_num = 0, 0
        for r in filtered_df["centered results"]:
            if not math.isnan(r):
                c_sum += r
                c_num += 1
        if c_num == 0:
            values_by_cat[c] = ""
            print(f"  {c} (no low-variance tasks in category)")
        else:
            c_avg = c_sum / c_num
            values_by_cat[c] = c_avg
            print(f"  {c} {c_avg}")

    return task_categories, values_by_cat


def row_label_for_path(json_path):
    return Path(json_path).stem


def format_cell(v):
    if v == "" or v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.5f}"
    return str(v)


def main():
    parser = gen_parser()
    args = parser.parse_args()

    eval_metadata = pd.read_csv(args.eval_meta_data)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(__file__).resolve().parent / "google_sheet.txt"

    rows_out = []
    column_order = None

    for json_path in args.eval_results:
        json_path = os.path.abspath(json_path)
        print(f"== {json_path} ==")
        with open(json_path, "r") as f:
            data = json.load(f)

        cats, values_by_cat = compute_row(data, eval_metadata)
        if column_order is None:
            column_order = ["Name"] + cats
        else:
            if cats != column_order[1:]:
                raise ValueError("Category column mismatch between JSON files")

        label = row_label_for_path(json_path)
        row_cells = [label] + [format_cell(values_by_cat[c]) for c in cats]
        rows_out.append(row_cells)

    lines = ["\t".join(column_order)]
    for row in rows_out:
        lines.append("\t".join(row))

    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"\nWrote {out_path} ({len(rows_out)} data rows + header)")


if __name__ == "__main__":
    main()
