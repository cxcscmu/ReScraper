# baselines/

Code for the baseline corpora in the main table (`tab:html2text`) and the scraper comparison (Figure 3,
`fig:scraper-comparison`), plus page-level runners of the same baselines used by the Section 5 analyses.

All baselines start from the same source pool. It is the DCLM `dclm-pool-400m-1x` raw HTML: a 10% sample of the
pages that DCLM's URL and language step already passed, 10,319 shards and 18,025,558 pages (`$HTML_POOL_DIR`).
Every model-based baseline runs the officially released model. The steps shared by every corpus are not repeated
here:
- BFF dedup (13-gram, threshold 0.8, `old-both`): `corpus/dedup_bff.sh`
- GPT-NeoX-20B tokenization: `corpus/tokenize.sh`
- DCLM pretraining and Core evaluation: `pretraining/`

The scripts call these with the corpus names listed below (see `corpus/corpora.sh`). DCLM itself is not vendored:
data processing uses the DCLM checkout with `pretraining/dclm_patches` applied. That checkout provides the
`resiliparse_extraction_modifier`, the RefinedWeb rule configs, `ray_processing/process_no_ray.py`, and the stub
`baselines/core/monitoring.py`.

| paper row / bar | pipeline | code | corpus name |
|---|---|---|---|
| RefinedWeb-rule (7.04B); Figure 3 resiliparse | resiliparse -> DCLM RefinedWeb rules -> BFF | `rule_based/run_resiliparse_pipeline.sh` | `refinedweb_rule` (READ `resiliparse_no_fasttext`) |
| C4-rule (5.06B) | resiliparse -> C4 rules | page level only: `rule_based/page_rules.py` (see below) | - |
| FineWeb-rule (5.16B) | resiliparse -> datatrove FineWeb filters | 60-shard sample and page level: `rule_based/fineweb_rule/` | - |
| ProX-C | resiliparse -> ProX chunk-level refining (`gair-prox/web-chunk-refining-lm`) -> BFF | `prox_c/` | `prox_c` (READ `prox_c_norule_noft`) |
| UltraX (10.78B, 400M/1B rows) | resiliparse (no filter) -> `openbmb/UltraX-0.6B-Preview` -> BFF | `ultrax/` | `ultrax` (READ `resiliparse_raw_ultrax_dedup_noft`) |
| UltraX (5.99B, 3B row) | RefinedWeb-rule corpus (after BFF) -> UltraX, no second dedup | `ultrax/` (other input) | `ultrax_rwrule` (READ `resiliparse_ultrax_norule_noft`) |
| Figure 3 trafilatura, jusText | scraper -> RefinedWeb rules -> BFF | page level only: `evaluation/extraction/extract_baselines.py` | - |
| Figure 3 Dripper (1B) | Dripper text -> RefinedWeb rules -> BFF | `scrapers/dripper_refinedweb_rule.sbatch` | `dripper_refinedweb_rule` (READ `dripper_refinedweb_rule_noft`) |
| Figure 3 Dripper (400M) | Dripper text -> BFF (no rules) | `corpus/dedup_bff.sh` on `$DRIPPER_STEP2_DIR` | `dripper_norule` |
| Figure 3 Dripper + UltraX | Dripper text -> UltraX -> BFF | `ultrax/dripper_ultrax/` | `dripper_ultrax` (READ `dripper_ultrax_noft`) |
| DataOrchestra | authors' released system | not included, see `dataorchestra/` | - |
| Raw data | uncleaned pool text (see the paper) | not included (see below) | - |

Dripper's own run over the pool (step 1 main-content inference, step 2 text) is in `data_construction/dripper/`.

The following code is not included in this release:
- the full-pool C4-rule and FineWeb-rule corpora;
- the full-pool trafilatura and jusText extraction runs;
- the tokenization of the Raw-data row;
- DataOrchestra.

The recipes of these runs are described in the paper. The resiliparse extraction stage of
`rule_based/run_resiliparse_pipeline.sh` writes the unfiltered resiliparse extraction of the pool (18,001,982
documents).

Environment variables used here (see `configs/paths.env.example`): `WORK_DIR`, `RESCRAPER_ROOT`, `DCLM_DIR`,
`HF_HOME`, `HTML_POOL_DIR`, `DRIPPER_STEP1_DIR`, `DRIPPER_STEP2_DIR`, `REFINER_PY`, `PIPE_PY`, `DATATROVE_PY`,
`LID_MODEL`, `GPU_PARTITION`, `CPU_PARTITION`, plus the per-directory variables listed in each README.
The `#SBATCH --partition=gpu|cpu` lines are placeholders. Override them with `-p` or `SBATCH_PARTITION`. Logs go to
`logs/` relative to the submission directory, so create it first.
