# baselines/prox_c/

ProX-C (Zhou et al., 2024; https://github.com/GAIR-NLP/ProX) is ProX's chunk-level refining program only. The
document-level program is not used. We run the released refining model `gair-prox/web-chunk-refining-lm` on
resiliparse-extracted text. The program executor is ProX's `utils/chunk_utils.py`, vendored unchanged as
`lib/prox_chunk_utils.py`. `trunc_text`, `merge_chunks` and the sampling parameters are copied from ProX's
`data_gen/tasks/apply_chunk_refining.py`.

Settings (identical in all runners):
- Chunking: up to 1,500 tokens per chunk, lines prefixed `[NNN]`.
- Prompt: `[doc]\n<chunk>\n[/doc]`. The checkpoint ships no chat template. With `PROXC_FMT=auto`, `proxc_pool.py`
  scores four candidate templates on real pages; every rank of our pool run selected this "plain" format.
- Decoding: greedy (temperature 0, top-p 0.9), 256 new tokens, `max_model_len` 2048. Chunks longer than that are
  skipped.
- Execution: `execute_meta_operations(threshold_1=0.0, threshold_2=0.95, error_op=2)`.

A page is dropped only when its program blanks it or fails to execute. No length, language or rule filter is
applied.

## Files
| file | what it does |
|---|---|
| `proxc_pool.py` | Pool runner, one process per GPU, `<rank> <world>`. The page universe is the Dripper step-1 records with non-empty `input` and `main_html`, the same pages the student reads (`lib/pool_join.py`). For each page: resiliparse extraction (60 s timeout, worker pool), then ProX-C. It writes `$INFER_OUT_DIR/<stem>_processed.jsonl.gz` rows `{"text"}`, atomically. It is resumable per shard. |
| `infer_pool_proxc.sbatch` | One array task = one node, 8 ranks. `sbatch --array=0-27 --export=ALL,INFER_OUT_DIR=$WORK_DIR/dclm_pipeline/prox_c/text,WORLD=224 infer_pool_proxc.sbatch` |
| `finish_proxc.sbatch` | Run with `--dependency=afterany:<array>`. See the section below. |
| `proxc_pages.py` | The same model and settings on a JSONL of pages `{k, text}`. Output: `{k, proxc, outcome, program}`. Used for page-level comparisons such as the case studies. |

### `finish_proxc.sbatch`
1. Backfills every step-1 shard that has no output: at most 28 more node tasks, then it re-submits itself in
   `nobf` mode.
2. Gates on the shard count: at least 10,315, or `EXPECT_SHARDS`.
3. Submits `corpus/dedup_bff.sh prox_c` and `corpus/tokenize.sh prox_c`.

Two notes:
- Two pool shards are excluded from every corpus in this repository (`POISON_STEMS`).
- The student post-filter is deliberately not applied, because ProX-C cannot emit operation syntax.

## Environment
- Paths: `WORK_DIR`, `RESCRAPER_ROOT`, `DRIPPER_STEP1_DIR`.
- Python: `REFINER_PY`, a vLLM 0.11.1 environment with `resiliparse` installed.
- Tunables: `PROXC_MODEL` (default `gair-prox/web-chunk-refining-lm`), `PROXC_FMT`, `PROXC_DTYPE`, `EXTRACT_PROCS`,
  and `INFER_STEMS_FILE` for backfill.
- In the sbatch scripts, `VLLM_CACHE_ROOT` is set per task to a node-local directory.

## Compute
- One node task has 8 GPUs (H200 in our runs) and handles about 370 shards of the 10,318; it took about
  16 minutes.
- The full pool takes 28 such tasks.
