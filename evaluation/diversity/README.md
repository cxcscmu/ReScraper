# evaluation/diversity/

Distinct n-gram share of each pipeline's output on the same 376 pool shards (Figure 6, right; data file
`diversity_ngrams_pool376.json`).

## Shards and corpora
`stems376.txt`: the 376 Common Crawl shard stems of the DCLM pool on which the pool-scale ProX-C run finished (all four
plotted corpora exist on each). Order: `random.Random(0).shuffle(sorted(stems))`, the same for every corpus.

| corpus | text (before deduplication) | default location (env override) |
|---|---|---|
| `rescraper` (plotted) | released ReScraper corpus after the post-filter (`inference/postfilter.py`) | `$WORK_DIR/dclm_pipeline/rescraper_corpus/text_clean` (`RESCRAPER_CORPUS_DIR`) |
| `ultrax` (plotted) | UltraX over the raw resiliparse extraction, column `cleaned`, non-empty | `$ULTRAX_RERUN_DIR/post` |
| `proxc` (plotted) | ProX-C over the raw resiliparse extraction | `$WORK_DIR/dclm_pipeline/prox_c/text` (`PROXC_POOL_DIR`) |
| `refinedweb_rule` (plotted) | resiliparse + DCLM language filter + RefinedWeb rules | `$WORK_DIR/dclm_pipeline/resiliparse/step3b_post_lang/dclm_baseline_refinedweb_post_lang/processed_data` (`RW_RULE_POOL_DIR`) |
| `fineweb_rule` | FineWeb-rule chain on the raw resiliparse text (`fw_scale.py`; no URL filter, the parquets carry no URL) | `$DIV_DIR/fw` |
| `resiliparse_raw` | raw resiliparse text (UltraX input), column `original` | `$ULTRAX_RERUN_DIR/post` |

## Counting (`ngram_scale.py`)
GPT-NeoX-20B tokenizer (`add_special_tokens=False`); n in {3, 5, 7}; n-grams within documents only; empty /
whitespace-only documents dropped. Distinct counting is exact over 64-bit keys (n <= 4: the token ids packed into one
uint64, collision-free; n > 4: chained splitmix64 hash), the union kept as a sorted uint64 array; no sketching. Per
corpus and cumulative shard count: distinct n-grams, tokens, documents, n-gram occurrences, and the sum of
per-document distinct n-grams (within- vs cross-document repeats). The figure plots distinct / occurrences at equal
token counts (`analysis/plot_quality_diversity.py`).

## Steps
1. `sbatch fw.sbatch` (CPU, 24 cores): `extract_raw.py` (raw text out of the UltraX parquets, `REFINER_PY` with pyarrow)
   -> `fw_scale.py 22` (FineWeb-rule, `DATATROVE_PY`; 665,603 documents in, 232,011 kept) -> `$DIV_DIR/fw/`.
2. `sbatch count.sbatch rescraper ultrax proxc refinedweb_rule fineweb_rule resiliparse_raw` (CPU, 24 cores, 400 GB;
   one process per corpus in parallel, 7-22 min each) -> `$DIV_DIR/results/<corpus>.json`.
3. `python build_json.py diversity_ngrams_pool376.json` -> the figure data file `diversity_ngrams_pool376.json` of `analysis/data`.

Environment: `RESCRAPER_ROOT`, `WORK_DIR`, `HF_HOME`, `DIV_DIR` (default `$WORK_DIR/eval/diversity`),
`RESCRAPER_CORPUS_DIR`, `ULTRAX_RERUN_DIR`, `PROXC_POOL_DIR`, `RW_RULE_POOL_DIR`, `REFINER_PY`, `DATATROVE_PY`,
`DCLM_DIR` (fastText LID model for FineWeb-rule). SLURM partitions are placeholders (`CPU_PARTITION` or `-p`).
