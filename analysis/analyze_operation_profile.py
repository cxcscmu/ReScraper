#!/usr/bin/env python3
"""Aggregate operation counts from materialized ReScraper JSONL.GZ shards.

Each retained row must contain ``e2e_tag``. Page deletions are inferred from the
known input-page count because deleted pages are intentionally absent from the
materialized corpus.

Corpus-side cross-check of the materialization block of Table 9 (tab:stage-retention): on the
post-filtered shards, the extract / refine / rewrite counts are the documents kept by <keep> /
<edit> / <rewrite> (the table itself reports the post-filter's own counters).

usage: analyze_operation_profile.py <shard_dir> <out.json> --input-pages N [--tokens T] [--workers W]
"""

import argparse
import gzip
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path


TAGS = ("<extract>", "<refine>", "<rewrite>")


def count_shard(path: Path) -> Counter:
    counts: Counter = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                counts["malformed_json"] += 1
                continue
            tag = row.get("e2e_tag", "<missing>")
            counts[tag] += 1
            counts["retained_characters"] += len(row.get("text", ""))
    counts["shards"] = 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--input-pages", type=int, required=True)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    shards = sorted(args.input_dir.glob("*_processed.jsonl.gz"))
    if not shards:
        raise SystemExit(f"no processed shards found under {args.input_dir}")

    total: Counter = Counter()
    with Pool(args.workers) as pool:
        for counts in pool.imap_unordered(count_shard, shards, chunksize=8):
            total.update(counts)

    retained = sum(total[tag] for tag in TAGS)
    unknown = sum(
        value
        for key, value in total.items()
        if key.startswith("<") and key not in TAGS
    )
    deleted = args.input_pages - retained - unknown
    if deleted < 0:
        raise RuntimeError(
            f"retained rows ({retained + unknown}) exceed input pages ({args.input_pages})"
        )

    result = {
        "input_pages": args.input_pages,
        "processed_shards": total["shards"],
        "counts": {
            "extract": total["<extract>"],
            "refine": total["<refine>"],
            "rewrite": total["<rewrite>"],
            "delete": deleted,
            "unknown": unknown,
        },
        "retained_characters": total["retained_characters"],
        "tokens": args.tokens,
    }
    result["shares"] = {
        key: value / args.input_pages for key, value in result["counts"].items()
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
