# analysis/

Scripts behind every figure and every data-driven table of the paper. Plotting scripts read small JSON summaries
("figure data files") and write PDFs; the analysis scripts compute those summaries (or the table numbers) from the
per-page outputs of `evaluation/`, the SFT sets of `data_construction/`, the inference logs and corpus directories
of `inference/` and `corpus/`, and SLURM accounting. Numbers hard-coded in a plotting script (Core scores, rule-group
shares, keep/drop shares) are the paper's numbers; the script that produces them is named in its docstring and below.

## Running

```bash
source configs/paths.env                     # WORK_DIR, RESCRAPER_ROOT, HF_HOME, ...
export FIG_DATA_DIR=$PWD/analysis/data       # figure data files (default: analysis/data next to the scripts)
export FIG_DIR=$PWD/analysis/figures         # output PDFs (default: analysis/figures), created on demand
python analysis/plot_quality_diversity.py    # any plot_*.py; no arguments needed
bash analysis/export_html_figure.sh          # Figure 2 (needs Chrome/Chromium and Ghostscript)
```

- Every `plot_*.py` reads `$FIG_DATA_DIR/<name>.json` (names in the map below) and writes `$FIG_DIR/<figure>.pdf`.
  Scripts with one data file accept `--data FILE` to read another one; several accept `--out FILE` (another output
  path) and `--png FILE` (a raster preview).
- `analyze_*.py` / `compute_*.py` that read per-page outputs need `WORK_DIR`; they write their JSON into
  `$FIG_DATA_DIR` by default (`--out` overrides). Per-page inputs live where `evaluation/` writes them:
  `$WORK_DIR/eval/heldout5k/heldout5k.jsonl` (the 5,000 held-out pages with the teacher program) and
  `$WORK_DIR/eval/heldout_pipeline/work/heldout5k_rel/{ours,sys_*}.jsonl` (the released model under release
  decoding and each baseline, page-aligned by `gid`).
- Other variables: `CORPUS_DIR` and `NSH` (`token_retention_sample.py`), `CHROME` (`export_html_figure.sh`), `HF_HOME`
  (GPT-NeoX-20B tokenizer for `compute_token_waterfall_5k.py`).
- The figure data files themselves are not part of the repository; the map below says which upstream step
  produces each one. The per-page judge files `keep_judge_5000.jsonl` and `rule_flags_5000.jsonl` will be released
  under `analysis/keep_judge/` of the `<HF_ORG>/<DATASET>` dataset upon acceptance.

All scripts are CPU-only. Everything runs in seconds to minutes on a laptop except `count_sft_tags.py` /
`compare_sft_tags.py` (stream the 1.38M-row Stage-1 set), `analyze_operation_profile.py` (all corpus shards,
multiprocessing) and `token_retention_sample.py` (60 sampled shards at three stages; the paper run was allocated
16 CPUs, 100 GB and a 2 h limit).

## Figure / table map

Every figure (Figure 2 is a drawing) and every data-driven table of the paper is listed. Tables 1, 4, 5 (method and
configuration), 10 and 12-16 (worked examples, case studies, prompt settings) are written by hand and have no script.
Paths such as `heldout5k_rel/ours.jsonl` are relative to `$WORK_DIR/eval/heldout_pipeline/work/`.

| Paper item | Script(s) in `analysis/` | Input data file | Produced upstream by |
|---|---|---|---|
| **Fig. 1(a)** `fig:motivation-rules`, "Rule drops" (share of each rule group's drops judged worth keeping) | `plot_rule_worth_keeping.py` (shares hard-coded) <- `analyze_rule_groups.py` | `rule_flags_5000.jsonl` | `evaluation/judges/`: gpt-oss-120b keep-or-drop judge on the held-out pages + per-rule flags of the RefinedWeb-rule / FineWeb-rule stacks (every rule evaluated without early stop) |
| **Fig. 1(b)** `fig:motivation-keepdrop`, "Keep--drop vs. downstream" | `plot_keepdrop_vs_core.py` (values hard-coded) <- `evaluation/judges/rule_motivation.py` (`keepdrop_fig1b` block) | `keep_judge_5000.jsonl`, `output_judge_5000.jsonl`, `heldout5k.jsonl`, `heldout5k_rel/ours.jsonl`, `heldout5k_rel/sys_{refinedweb_rule,fineweb_rule,proxc,ultrax}.jsonl`; Core = 1B rows of Table 2 | `evaluation/judges/` (judge), `evaluation/heldout_set/` (held-out pages), `evaluation/heldout_pipeline/` (per-page outputs of ReScraper and the baselines); Core: see Table 2 |
| **Fig. 2** `fig:method`, "Overview of ReScraper" | `export_html_figure.sh` | `figure_sources/method_overview.html` | drawing, no data |
| **Table 2** `tab:html2text`, "Benchmarking pretraining data curation methods" | none (procedure below) | Core TSV per pretraining run; tokenized-dataset JSON per corpus | `pretraining/pretrain_400m.sh`, `pretraining/pretrain_1b.sh` (and the 3B setting), evaluated with `pretraining/core_sheet.sh` -> `pretraining/eval_sheet.py`; `#Unique Tokens` = `num_tokens` written by `corpus/tokenize.sh`; corpora from `baselines/` (rule stacks, ProX-C, UltraX, DataOrchestra) and `inference/` + `corpus/` (ReScraper) |
| **Fig. 3** `fig:scraper-comparison`, "Core score ... for each heuristic scraper" | `plot_scraper_comparison.py` (Core hard-coded) | none | Core TSVs as for Table 2, for the `baselines/scrapers/` corpora (resiliparse, trafilatura, jusText, Dripper, each + RefinedWeb-rule) and Dripper + UltraX (`baselines/ultrax/`) |
| **Table 3** `tab:ablation-supervision`, "Ablation of the operations of ReScraper" | none (procedure below) | Core TSVs, tokenized-dataset JSONs | arms from `ablations/operation_ablation/` (one inference pass, one arm per disabled operation), "Dripper `<extract>`" row from `ablations/two_stage_refiner/`; then `corpus/` and `pretraining/` as for Table 2 |
| **Fig. 4** `fig:operation-scores`, "Mean quality scores of pages before and after each operation" | `plot_operation_scores.py` | `operation_scores.json` | `evaluation/heldout_pipeline/`: build stage (`s4_build.py`) over the DataMan and FineWeb-Edu scores of the release-decoding outputs |
| **Fig. 5** `fig:operation-mix`, "Share of pages per operation for each model-based pipeline" | `plot_operation_mix.py` | `operation_mix.json` | `evaluation/heldout_pipeline/` build stage (per-page operation classes of ReScraper, UltraX, ProX-C); the DataOrchestra counts on the same 5,000 pages were computed separately from the DataOrchestra run (`baselines/dataorchestra/`) and added to the file |
| **Fig. 6** `fig:quality-diversity`, "mean quality score after cleaning ... distinct n-gram share" | `plot_quality_diversity.py` | `quality_buckets.json`, `diversity_ngrams_pool376.json` | `evaluation/heldout_pipeline/` build stage (quality buckets); `evaluation/` diversity step (distinct n-grams of each corpus on the same 376 pool shards; `ngram_scale.py` + `build_json.py`) |
| **Fig. 7** `fig:decision-flow`, "Operations chosen by the teacher and by ReScraper" | `plot_decision_sankey.py` | `decision_flow.json` | `evaluation/heldout_pipeline/`: release-decoding execution stage (`s1_rel_exec.py`), teacher tag of `heldout5k.jsonl` x ReScraper decision |
| **Fig. 8** `fig:extraction`, "Token F1 against Dripper ... gpt-oss-120b score without a reference" | `plot_extraction.py` | `extraction_quality.json` = `{"ext_vs_dripper": ..., "judge": ...}` | `evaluation/` extraction evaluation: token P/R/F1 against Dripper (`ext_vs_dripper_5000.json`, `ext_metrics.py`) and the gpt-oss-120b extraction-judge summary (`ext_judge_5000.json`, `agg_extract.py`); `--ext` / `--judge` read the two files directly |
| **Table 6** `tab:sft-composition`, "SFT targets per operation in each training stage" | `count_sft_tags.py` (Stage 1, `first_tag`), `analyze_sft_operations.py` (Stage 2, `decision:<tag>`), `compare_sft_tags.py` (check against the no-rewrite base set) | Stage-1 and Stage-2 SFT JSONL | `data_construction/sft_sets/build_stage1_set.py`, `build_stage2_set.py` (base set: `data_construction/base_set/`); "fit the 32,768-token sequence" = the `train/validation X -> Y` filter counts printed by `training/prepack_sft.py` |
| **Table 7** `tab:teacher-cost`, "Cost in H200 GPU hours from Slurm accounting" | `slurm_gpu_hours.py`, `prorate_gpu_hours.py` | `sacct` records; per-shard Dripper stats (see script) | jobs of `data_construction/dripper/`, `teacher_refine/`, `rescue/rewrite_pool.*`, `training/stage1_chain.sbatch`, `stage2_chain.sbatch`, `inference/infer_pool.sbatch`, `baselines/`, `pretraining/pretrain_1b.sh` (procedure below) |
| **Table 8** `tab:fidelity-by-length`, "Teacher fidelity ... split into equal quartiles by input length" | `analyze_fidelity_by_length_5k.py` (`--tex` writes the tabular; the table is convention `main_text`) | `heldout5k.jsonl`, `heldout5k_rel/ours.jsonl` -> `fidelity_by_length_5k.json` | `evaluation/heldout_set/`, `evaluation/heldout_pipeline/` |
| **Fig. 9** `fig:token-waterfall`, "Tokens of the 4,989 held-out pages ... left after each stage" | `compute_token_waterfall_5k.py` -> `plot_token_waterfall_5k.py` | `token_waterfall_5k.json` (computed here from `heldout5k.jsonl` x `heldout5k_rel/ours.jsonl`) | `evaluation/heldout_set/`, `evaluation/heldout_pipeline/` |
| **Table 9** `tab:stage-retention`, "Per-stage retention" | `stage_retention_from_logs.py` (rendering and routing blocks), `token_retention_sample.py` (token axis), `analyze_operation_profile.py` (cross-check of the materialization block) | pool-inference logs; `text/`, `text_clean/`, `step4_dedup_oldboth_65b/` of the corpus directory | `inference/infer_pool.py` (per-shard counter lines), `inference/postfilter.py` (`POSTFILTER_STATS` line = materialization block), `corpus/dedup_bff.sh`; "Released corpus" = `num_tokens` from `corpus/tokenize.sh` |
| **Fig. 10** `fig:length-dist`, "Document length in GPT-NeoX-20B tokens" | `plot_length_distribution.py` | `dist_length.json` | `evaluation/heldout_pipeline/` build stage (token counts of every kept output) |
| **Table 11** `tab:operation-stats`, "Operations of ReScraper on the 4,989 held-out pages" | `analyze_operation_stats_5k.py` (`--tex` writes the tabular) | `heldout5k.jsonl`, `heldout5k_rel/ours.jsonl` -> `operation_stats_5k.json` | `evaluation/heldout_set/`, `evaluation/heldout_pipeline/` |

## Procedures without a dedicated script

**Tables 2 and 3 (Core and #Unique Tokens).** Each row of a setting is one pretraining run: tokenize the deduplicated
corpus (`corpus/tokenize.sh`; its dataset JSON's `num_tokens` is the `#Unique Tokens` column), pretrain
(`pretraining/pretrain_400m.sh`, `pretraining/pretrain_1b.sh`), evaluate, and run `pretraining/core_sheet.sh <run>`.
The category columns and Core are read from the TSV that `pretraining/eval_sheet.py` writes (centered accuracy,
mean over the 22 low-variance tasks with `commonsense_qa` dropped); do not read Core from a field of the metrics
JSON. Bold / underline mark the best / second-best value within each scale.

**Table 7 (GPU hours).** Every row except ProX-C and DataOrchestra is `sacct -D -X` accounting with all attempts
counted (`slurm_gpu_hours.py`):
`--name dripper_step1`, `--name teacher_refine`, `--name rewrite_pool`, `--name-prefix stage1_`, `--name stage2`,
`--submit-contains INFER_OUT_DIR=$WORK_DIR/dclm_pipeline/rescraper_corpus/text` (main inference array and every
backfill job of the released corpus), the UltraX pool jobs of `baselines/ultrax/`, and `--name-prefix pt1b_` (the
1.4B pretraining row is the range over corpora). Teacher rows count only the share spent on the pool shards the SFT
set is drawn from: Dripper = its full-pool run pro-rated by per-shard minutes (`prorate_gpu_hours.py`, "global
pro-rata by minutes"), Qwen3.8-27B = its single labelling job (it ran on the seed pages only), RePro = its pool job
times the share of its rewrite candidates in the SFT shards. Stage 1 is counted up to the epoch-2 checkpoint that
Stage 2 continues from. The ProX-C and DataOrchestra pool rows are not computed by these scripts.

**Table 9 (stage retention).** `stage_retention_from_logs.py 'logs/rescraper_infer_*.log'` (plus the backfill logs)
prints the summed per-shard counters: records covered by the logs = `pages` plus the records skipped before
rendering (`no_main_html`; unreadable records: `empty_step1_input`, `bad_step1_json`), pages rendered = `pages`,
omitted = `render_fail_or_empty` and `too_long`, routed = `pages - render_fail_or_empty - too_long`, routing = the
`tag<...>` counters and `parse_fail`. The materialization block is the `POSTFILTER_STATS` line of `inference/postfilter.py`
(`in`, `dropped`, `kept`, `tag_<...>`). The token axis is `token_retention_sample.py` (60 sampled shards, characters
divided by the measured 4.2601 characters per GPT-NeoX-20B token).

## Scripts

| script | what it does |
|---|---|
| `plot_rule_worth_keeping.py`, `analyze_rule_groups.py` | Fig. 1(a); merges first-failing rules of both stacks into five groups and prints each group's worth-keeping share |
| `plot_keepdrop_vs_core.py` | Fig. 1(b); keep-drop accuracy = mean of worth-keeping pages kept and junk pages removed (dropped, or kept only as text the output judge rates worth keeping), from the `keepdrop_fig1b` block of `evaluation/judges/rule_motivation.py` |
| `export_html_figure.sh`, `figure_sources/method_overview.html` | Fig. 2; headless Chrome prints the SVG page, Ghostscript crops it to the ink box |
| `plot_scraper_comparison.py` | Fig. 3 |
| `plot_operation_scores.py`, `plot_operation_mix.py` | Figs. 4 and 5 (same canvas, placed side by side) |
| `plot_quality_diversity.py` | Fig. 6 (`--with-fineweb` adds FineWeb-rule, not in the paper) |
| `plot_decision_sankey.py`, `plot_extraction.py` | Figs. 7 and 8 |
| `compute_token_waterfall_5k.py`, `plot_token_waterfall_5k.py` | Fig. 9; per-page cumulative stage tokens (input rendering, after extract / delete / edit / rewrite) |
| `plot_length_distribution.py` | Fig. 10; log-space Gaussian KDE per pipeline, x axis cut at 2,000 tokens |
| `analyze_fidelity_by_length_5k.py` | Table 8; decision accuracy and pooled word-token P/R/F1 against the teacher per input-length quartile, three page-set conventions |
| `analyze_operation_stats_5k.py` | Table 11; `<extract>` and `<edit>` program statistics of the model and, with the same code, of the teacher |
| `count_sft_tags.py`, `compare_sft_tags.py`, `analyze_sft_operations.py` | Table 6 |
| `slurm_gpu_hours.py`, `prorate_gpu_hours.py` | Table 7 |
| `stage_retention_from_logs.py`, `token_retention_sample.py`, `analyze_operation_profile.py` | Table 9 |

## Environment

- Python 3.10 or newer. The paper figures were drawn with matplotlib 3.7.5 (Figs. 1, 3-8) and matplotlib 3.10.8 with
  numpy 2.2.6 and scipy 1.17.1 (Figs. 9-10; the `REFINER_PY` environment of `requirements/refiner.txt` has these).
  `plot_length_distribution.py` needs scipy (`gaussian_kde`); no script needs seaborn or pandas.
  `compute_token_waterfall_5k.py` needs `transformers` (GPT-NeoX-20B tokenizer) and `lib/`; run it with `REFINER_PY`.
- Fonts: serif, with the fallback chain Times New Roman -> Times -> Nimbus Roman -> TeX Gyre Termes -> DejaVu Serif
  and STIX for math. The paper PDFs of Figs. 1, 3-8 embed Times New Roman, those of Figs. 9-10 Nimbus Roman (URW base35,
  the Times metric clone on Linux). `plot_rule_worth_keeping.py` and `plot_keepdrop_vs_core.py` set
  `font.family = "Times New Roman"` with no fallback: install Times New Roman (e.g. the msttcorefonts package) or
  matplotlib substitutes DejaVu Sans with a `findfont` warning. Clear the matplotlib font cache after installing fonts.
- `export_html_figure.sh`: Chrome or Chromium (headless), Ghostscript and python3; the page loads IBM Plex Sans / Mono
  from Google Fonts, so exporting needs network access or locally installed IBM Plex fonts.
