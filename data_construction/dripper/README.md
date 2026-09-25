# data_construction/dripper/

The Dripper (MinerU-HTML) teacher is run over the whole source pool. Its outputs are used in several places:
- the SFT targets (extraction part of every Stage-1 target);
- the page universe of the student's pool inference and of the ProX-C runner (`lib/pool_join.py`);
- the Dripper bars of Figure 3 (`baselines/scrapers/`, `baselines/ultrax/dripper_ultrax/`).

Dripper is third-party and not vendored:
- Python package `dripper` 1.0.0 from https://github.com/opendatalab/MinerU-HTML.
- Model `opendatalab/MinerU-HTML` (HF), run with its vLLM backend.
- The text renderer is `webpage_converter` from `mineru-webkit` 0.1.6, a dependency of `dripper`; it is wrapped in
  `lib/render.py`.

## Steps
| step | file | input -> output |
|---|---|---|
| 1 main-content inference | `step1_dripper_inference.py` (`run_shard`), launched by `run_dripper_pool.sbatch` | raw HTML pool shard `$HTML_POOL_DIR/<stem>.jsonl.gz` (`text` = raw HTML) -> `$DRIPPER_STEP1_DIR/<stem>.jsonl` with one row per page `{"input": raw html, "main_html": ...}` (`main_html` empty when Dripper found no main content) |
| 2 main_html -> text | `convert_main_html_to_text.py`, launched by `run_convert_text.sbatch` | `$DRIPPER_STEP1_DIR/<stem>.jsonl` -> `$DRIPPER_STEP2_DIR/<stem>.jsonl.gz` rows `{"text"}` |

Step 1 details:
- Pages are sorted by HTML length inside a shard before inference, so step-1 row order differs from pool order.
- Batches are capped at 500 pages and 5M characters.
- Dripper runs with `use_fall_back=True` and `raise_errors=False`. vLLM uses `gpu_memory_utilization` 0.95 and
  `max_num_seqs` 2048 (`DRIPPER_GPU_MEM`, `DRIPPER_MAX_SEQS`, `DRIPPER_BACKEND`).
- Array task *i* of `run_dripper_pool.sbatch` owns 64 consecutive shards of the sorted pool and runs one worker per
  GPU.
- A shard is marked `.done` after success, so requeued tasks skip it.

Step 2 details:
- Conversion is `render.webkit_txt(main_html)`, the renderer the student reads, with a 60 s per-page timeout
  (`DOC_TIMEOUT`).
- Pages with empty `main_html`, failed or timed-out conversion, or empty text are dropped. Rows otherwise keep
  step-1 order, and `lib/pool_join.py` re-aligns them.
- The step runs on CPUs and can run while step 1 is still going: existing outputs are skipped.
- When re-running a subset of array indices, pass the original task count as `NT`.

Downstream corpora built from these outputs:
- Dripper + RefinedWeb-rule: `baselines/scrapers/dripper_refinedweb_rule.sbatch`.
- Dripper without rules: `corpus/dedup_bff.sh dripper_norule`.
- Dripper + UltraX: `baselines/ultrax/dripper_ultrax/`.

## Usage
```bash
source configs/paths.env; mkdir -p logs
sbatch --array=0-161 data_construction/dripper/run_dripper_pool.sbatch     # 162 x 64 shards
sbatch --array=0-15  data_construction/dripper/run_convert_text.sbatch
```

## Environment
- Paths: `WORK_DIR`, `RESCRAPER_ROOT`, `HTML_POOL_DIR`, `DRIPPER_STEP1_DIR`, `DRIPPER_STEP2_DIR`.
- Model: `HF_HOME`, or `DRIPPER_MODEL` pointing to a local MinerU-HTML snapshot.
- Python: `REFINER_PY`, i.e. vLLM 0.11.1 + `dripper` 1.0.0 + bs4.

## Compute
- Step 1: one node with 8 GPUs (H200 in our runs) per array task, 12 h limit, resumable. The share of the Dripper
  run spent on the SFT pages is reported in the paper's cost table (149.6 GPU-h).
- Step 2: CPU only; 16 tasks of 32 CPUs each.
