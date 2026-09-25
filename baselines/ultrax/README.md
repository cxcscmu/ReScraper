# baselines/ultrax/

UltraX (Zhao et al., 2026) is run with its officially released model and scripts:
- Model: `openbmb/UltraX-0.6B-Preview`, subfolder `models/UltraX` of the HF snapshot.
- Code: https://github.com/OpenBMB/UltraX. Our checkouts are at commit `cb0c2fa`.

Inference uses `stage2_large_scale_execution/inference/inference.py --num_gpus 8 --max_chars 48000`, with greedy
decoding hard-coded in that script. Post-processing uses
`stage2_large_scale_execution/post_processing/post_process_and_execute.py`, which executes the predicted cleaning
functions and writes the columns `original, cleaned, processed_functions`. UltraX is third-party and not vendored.
Point `ULTRAX_DIR` at a checkout.

## Corpora
| corpus | input (`UX_SRC`) | after UltraX | paper |
|---|---|---|---|
| `ultrax` (READ `resiliparse_raw_ultrax_dedup_noft`) | raw resiliparse extraction of the whole pool (see note 1) | pages UltraX emptied are dropped (13,686,451 documents), then BFF dedup (12,285,075 documents) and tokenization | main table UltraX, 10.78B unique tokens (400M and 1B rows); Section 5 UltraX samples; cost table (180 H200 GPU-h, all attempts) |
| `ultrax_rwrule` (READ `resiliparse_ultrax_norule_noft`) | the RefinedWeb-rule corpus after BFF dedup (see note 2) | emptied pages dropped (7,055,124 documents), tokenized without a second dedup | main table UltraX, 5.99B (3B row) |
| `dripper_ultrax` (READ `dripper_ultrax_noft`) | Dripper step-2 text, pages of at least 10 characters (`data_construction/dripper/`) | emptied pages and pages under 50 characters dropped, then BFF dedup | Figure 3, Dripper + UltraX |

Notes:
1. The `ultrax` input is `rule_based/run_resiliparse_pipeline.sh` phase 1: 10,319 shards and 18,001,982 pages.
   It has no length, language or rule filter.
2. The `ultrax_rwrule` input is `$WORK_DIR/dclm_pipeline/resiliparse/step4_dedup_oldboth_65b`, the text the
   7.04B RefinedWeb-rule row was tokenized from. It has already been through the length filter, the full RefinedWeb
   rule stack and BFF.

## Files
Resiliparse text -> UltraX (`ultrax` and `ultrax_rwrule`):

| file | stage |
|---|---|
| `submit_ultrax.sh` | Checks the input shard count, then submits `ux_prep` (16 CPU tasks) -> `ux_inf` (28 x 8 GPUs) -> `ux_gate` |
| `ux_pool_prep.py`, `ux_prep.sbatch` | `$UX_SRC/*.jsonl.gz` -> `$UX_WORK/in/<stem>.parquet`, column `original`. A row is kept iff `text.strip()` is non-empty; the text is not stripped. Resumable, atomic writes. |
| `ux_slice_in.py`, `ux_inf.sbatch` | Array task *i* symlinks the striped slice `files[i::28]` into `in_parts/<i>`, runs UltraX inference into `inf/`, then post-processing of its own shards into `post/`. Resumable. |
| `ux_validate.py` | Checks that `in`, `inf` and `post` row counts are equal per shard. `--clean` deletes truncated outputs, and the script writes the missing-shard lists. A completed inference task does not imply a complete slice, because a dead worker does not fail the task. |
| `ux_fix.sbatch` | Re-runs inference and post-processing for whatever is missing |
| `ux_gate.sbatch` | Runs after the array (`afterany`). Validates, then either submits a fix round (up to 3) or the finishing chain `ux_leg` -> `corpus/dedup_bff.sh $UX_CORPUS` (only if `UX_DEDUP=1`) -> `corpus/tokenize.sh $UX_CORPUS` |
| `ux_build_leg.py`, `ux_leg.sbatch` | `post/*.parquet` -> `text/<stem>.jsonl.gz` with `{"text": cleaned.strip()}`. Empty pages are dropped. Run `gzip -t` on the shards after any abnormal termination before deduplicating. |

Dripper text -> UltraX (`dripper_ultrax/`):

| file | stage |
|---|---|
| `dripper_ultrax_prep.py`, `.sbatch` | Dripper step-2 shards -> parquet in 6 contiguous parts; rows under 10 characters are dropped; damaged gzip shards are tolerated |
| `dripper_ultrax_todo.py`, `dripper_ultrax_infer.sbatch <part>` | Per part: lists missing or corrupt outputs, runs UltraX inference, then post-processing (`POSTPROC=0` defers it to `dripper_ultrax_post.sbatch <part>`) |
| `dripper_ultrax_rebuild.py`, `.sbatch` | `post` -> `text/<stem>.jsonl.gz`; drops emptied pages and pages under 50 characters; writes `rebuild_stats.json`. Then run `corpus/dedup_bff.sh dripper_ultrax` and `corpus/tokenize.sh dripper_ultrax`. |

## Usage
```bash
source configs/paths.env; export ULTRAX_DIR=/path/to/UltraX
# UltraX row (10.78B): input = raw resiliparse extraction (defaults)
bash baselines/ultrax/submit_ultrax.sh
# earlier 5.99B run: input = RefinedWeb-rule corpus after dedup, no second dedup
UX_SRC=$WORK_DIR/dclm_pipeline/resiliparse/step4_dedup_oldboth_65b UX_WORK=$WORK_DIR/dclm_pipeline/resiliparse_ultrax \
  UX_CORPUS=ultrax_rwrule UX_DEDUP=0 bash baselines/ultrax/submit_ultrax.sh
# Dripper + UltraX
sbatch baselines/ultrax/dripper_ultrax/dripper_ultrax_prep.sbatch
for k in 0 1 2 3 4 5; do sbatch baselines/ultrax/dripper_ultrax/dripper_ultrax_infer.sbatch $k; done
sbatch baselines/ultrax/dripper_ultrax/dripper_ultrax_rebuild.sbatch   # then corpus/dedup_bff.sh + tokenize.sh dripper_ultrax
```

## Environment
- Paths and model: `WORK_DIR`, `RESCRAPER_ROOT`, `HF_HOME` (or `ULTRAX_MODEL`), `ULTRAX_DIR`.
- Python: `REFINER_PY`, a vLLM 0.11.1 environment that runs the UltraX scripts.
- Tunables: `UX_SRC`, `UX_WORK`, `UX_CORPUS`, `UX_DEDUP`, `EXPECT_SHARDS`, `DRIPPER_UX_WORK` (Dripper + UltraX work dir),
  `GPU_PARTITION`, `CPU_PARTITION`.

## Compute
UltraX over the raw pool used 28 array tasks of 8 GPUs (H200). Most tasks finished in about 40 minutes, and the
run took 180.0 GPU-h including fix rounds and interrupted attempts. The earlier 5.99B run took about 50 GPU-h on its
smaller input. Preparation, validation and text building are CPU jobs of minutes.
