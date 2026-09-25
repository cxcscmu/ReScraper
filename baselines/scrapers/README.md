# baselines/scrapers/

The scraper comparison (Figure 3, `fig:scraper-comparison`) runs each heuristic scraper (resiliparse, trafilatura,
jusText) and Dripper over the same source pool. Each is followed by the same RefinedWeb-rule cleaning (DCLM
`dclm_baseline_refinedweb_post_lang.yaml`), then BFF dedup and tokenization.

| bar | code |
|---|---|
| resiliparse | `baselines/rule_based/run_resiliparse_pipeline.sh` (the same corpus as the RefinedWeb-rule row) |
| trafilatura, jusText | full-pool runs not included (see below); page-level runner `evaluation/extraction/extract_baselines.py` |
| Dripper, 1B | `dripper_refinedweb_rule.sbatch` on the Dripper step-2 text (`data_construction/dripper/`), corpus `dripper_refinedweb_rule` |
| Dripper, 400M | Dripper step-2 text deduplicated without the rule stack: `sbatch corpus/dedup_bff.sh dripper_norule`, then `corpus/tokenize.sh dripper_norule` |
| Dripper + UltraX | `baselines/ultrax/dripper_ultrax/` |

## Files
- `dripper_refinedweb_rule.sbatch` runs the Dripper leg:
  1. Dripper step-2 text -> DCLM `process_no_ray.py` with the RefinedWeb rule config ->
     `$WORK_DIR/dclm_pipeline/dripper/step3b_post_lang`.
  2. It then submits `corpus/dedup_bff.sh` and `corpus/tokenize.sh` for corpus `dripper_refinedweb_rule`.

  The steps mirror `run_resiliparse_pipeline.sh`, so only the extractor differs. It is resumable through a
  `.marks/step3b.done` marker. It needs one 32-CPU job; the rules take a few hours.
  Env: `WORK_DIR`, `DCLM_DIR` (with `pretraining/dclm_patches`), `RESCRAPER_ROOT`, `PIPE_PY`,
  `DRIPPER_STEP2_DIR`.
- `evaluation/extraction/extract_baselines.py`: trafilatura 2.0.0 (`trafilatura.extract(html)`, default settings, plain text) and
  jusText 3.0.2 (English stop list, default thresholds, non-boilerplate paragraphs joined by newlines) on the raw
  HTML of individual pages.
  - It uses a 120 s per-page timeout; failures give an empty string plus an error field.
  - Input rows are `{gid, html}`.
  - Used for the extraction-quality comparison on held-out pages (Figure `fig:extraction`).
  - CPU only; `REFINER_PY` has both packages.

## Not included
The full-pool trafilatura and jusText extraction runs used DCLM mappers (`trafilatura_extraction_modifier` and a
jusText extractor) written for those runs. That code is not part of this release; the paper describes their recipe.
