"""Dripper (MinerU-HTML) main-content inference over one shard of raw HTML (step 1 of the Dripper teacher run).

Reads a (gzipped) JSONL shard of raw HTML (field "text"), runs the Dripper main-content model on every page and writes
one row per page, in the order below, to <output_dir>/shard_<shard_id>.jsonl:
    {"input": <raw html>, "main_html": <the HTML region Dripper selected; "" when it found none>}
Rows are sorted by HTML length (short pages first) before inference, so the output row order is NOT the pool order;
downstream code joins rows by content (see lib/pool_join.py). A batch that raises is written with empty output
({"input": ..., "output": ""}) so that no page is silently lost.

Dripper is third-party (package `dripper` 1.0.0 from the MinerU-HTML repository, model opendatalab/MinerU-HTML) and is
not vendored here.

usage (1 GPU, selected with CUDA_VISIBLE_DEVICES; called by run_dripper_pool.sbatch):
    python step1_dripper_inference.py run_shard --shard_file <pool shard .jsonl.gz> --output_dir <step1 dir> \
        --model_path <local MinerU-HTML snapshot> --shard_id <global shard index> --batch_size 500
env: DRIPPER_BACKEND (default vllm), DRIPPER_GPU_MEM (default 0.95), DRIPPER_MAX_SEQS (default 2048)
"""

import argparse
import os
import gzip
import json
import time

from pathlib import Path


def cmd_run_shard(args):
    """Run Dripper inference on a single shard file."""
    from dripper.api import Dripper

    # Load shard data (small, only this shard's portion)
    print(f"[Shard {args.shard_id}] Loading shard from {args.shard_file}")
    records = []
    _opener = gzip.open if args.shard_file.endswith(".gz") else open
    with _opener(args.shard_file, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[Shard {args.shard_id}] Loaded {len(records)} records")

    # Load Dripper model (1 GPU via CUDA_VISIBLE_DEVICES)
    print(f"[Shard {args.shard_id}] Loading Dripper model from {args.model_path}")
    dripper = Dripper(config={
        'model_path': args.model_path,
        'use_fall_back': True,
        'raise_errors': False,
        'inference_backend': os.environ.get("DRIPPER_BACKEND", "vllm"),
        # max_num_seqs 2048 + gpu memory 0.95 gave the best docs/s on an H200
        'model_init_kwargs': {
            'gpu_memory_utilization': float(os.environ.get("DRIPPER_GPU_MEM", "0.95")),
            'max_num_seqs': int(os.environ.get("DRIPPER_MAX_SEQS", "2048")),
        },
    })

    # Process in batches
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"shard_{args.shard_id:03d}.jsonl"

    kept = 0
    no_main = 0
    t0 = time.time()

    # Sort records by HTML length (short first -> faster early batches)
    records.sort(key=lambda r: len(r.get("text", "")))

    with open(output_path, "w", encoding="utf-8") as f_out:
        i = 0
        batch_num = 0
        while i < len(records):
            # Adaptive batch size: limit total chars per batch to avoid
            # GPU OOM and reduce padding waste on long HTML
            max_batch_chars = 5_000_000  # ~5M chars per batch
            batch = []
            batch_chars = 0
            while i < len(records) and len(batch) < args.batch_size:
                html_len = len(records[i].get("text", ""))
                if batch_chars + html_len > max_batch_chars and len(batch) > 0:
                    break
                batch.append(records[i])
                batch_chars += html_len
                i += 1

            html_texts = [rec["text"] for rec in batch]
            batch_num += 1

            bt0 = time.time()
            try:
                results = dripper.process(html_texts)
            except Exception as e:
                print(f"[Shard {args.shard_id}] Batch {batch_num} Dripper error: {e}")
                # Still write empty outputs for failed batch
                for rec in batch:
                    out = {"input": rec["text"], "output": ""}
                    f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                    kept += 1
                    no_main += 1
                continue
            bt1 = time.time()

            for rec, result in zip(batch, results):
                main_html = result.main_html or ""
                if not main_html:
                    no_main += 1

                out = {"input": rec["text"], "main_html": main_html}
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1

            elapsed = time.time() - t0
            rate = kept / elapsed if elapsed > 0 else 0
            avg_chars = batch_chars // len(batch) if batch else 0
            print(f"[Shard {args.shard_id}] {i}/{len(records)} | "
                  f"kept={kept} no_main={no_main} | "
                  f"{rate:.1f} pairs/s | "
                  f"batch#{batch_num} n={len(batch)} avg={avg_chars//1000}k chars {bt1-bt0:.1f}s | "
                  f"{elapsed/60:.1f}min elapsed")

    elapsed = time.time() - t0
    print(f"\n[Shard {args.shard_id}] Done. {kept} pairs -> {output_path} "
          f"({no_main} with empty output, {elapsed/60:.1f}min)")

    # Write shard stats
    stats_path = output_dir / f"stats_shard_{args.shard_id:03d}.json"
    with open(stats_path, "w") as f:
        json.dump({
            "shard_id": args.shard_id,
            "num_samples": kept,
            "num_no_main": no_main,
            "elapsed_s": elapsed,
            "pairs_per_s": kept / elapsed if elapsed > 0 else 0,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run_shard command
    p_run = subparsers.add_parser("run_shard", help="Run Dripper on one shard")
    p_run.add_argument("--shard_file", type=str, required=True)
    p_run.add_argument("--output_dir", type=str, required=True)
    p_run.add_argument("--model_path", type=str, required=True)
    p_run.add_argument("--shard_id", type=int, required=True)
    p_run.add_argument("--batch_size", type=int, default=500)

    args = parser.parse_args()

    if args.command == "run_shard":
        cmd_run_shard(args)


if __name__ == "__main__":
    main()
