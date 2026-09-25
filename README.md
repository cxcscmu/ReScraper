# ReScraper: Unified Scraping and Cleaning of Web Data for Effective LLM Pretraining

Anonymous code release accompanying the ICLR 2027 submission. The trained model, the refined corpus and the SFT
data will be released upon acceptance (referred to below as `<HF_ORG>/<MODEL>` and `<HF_ORG>/<DATASET>`).

ReScraper replaces the heuristic HTML-to-text stack of pretraining pipelines (a rule-based scraper followed by
rule-based cleaning filters) with one small language model (Qwen3-0.6B). The model reads the visible text of a
raw web page, rendered one block per line with line ids `<lid:n>`, and generates a short program:

```
<keep>|<edit>|<delete>|<rewrite>      the operation chosen for the page
<extract>                             always first: remove boilerplate lines
rm A-B / rm A                         (lines that are not main content)
<same operation again>
payload                               <edit>: rm A-B / sub N: "s" (delete lines / strings); <rewrite>: new text
```

A deterministic executor (`lib/rescraper_ops.py`) applies the program to the rendered lines, so kept text is copied, not
regenerated. The supervision is built from three teachers run in sequence on the same rendering: Dripper
(main-content extraction -> `<extract>`), Qwen3.8-27B under a strict-subset refinement prompt (-> `<keep>`,
`<edit>`, `<delete>`), and the RePro 1B rephraser for teacher-deleted pages whose FineWeb-Edu score is at least 1.0
(-> `<rewrite>`). The student is fine-tuned in two stages (all operations; then a rewrite-heavy mixture), applied to
the raw HTML of the whole DCLM source pool, and the output is deduplicated and tokenized for DCLM pretraining at the
400M, 1B and 3B scales.

## Pipeline

```
DCLM pool sample (raw HTML, ~10.3K shards, 18.0M pages)
 │
 ├─ data_construction/dripper/        Dripper over the pool (teacher 1; also the base of the scraper baselines)
 ├─ data_construction/seed_pages/     ~1.6M SFT seed pages (held-out shards excluded by stem)
 ├─ data_construction/teacher_refine/ Qwen3.8-27B refinement labels (teacher 2)
 ├─ data_construction/base_set/       render raw HTML -> <lid:n> input; serialized target; no-rewrite base set (1.38M)
 ├─ data_construction/rescue/         FineWeb-Edu of deleted pages; RePro 1B rewrites (teacher 3)
 ├─ data_construction/sft_sets/       Stage-1 set (1.38M) and Stage-2 set (131K, 30% <rewrite>)
 ├─ training/                         Stage 1 (Qwen3-0.6B, 3 epochs) -> Stage 2 (from the epoch-2 checkpoint, 3 epochs)
 ├─ inference/                        Stage-2 model over the pool (T=1.0, 3,072 new tokens) -> executor -> post-filter
 ├─ corpus/                           BFF dedup (13-gram, 0.8) -> GPT-NeoX-20B tokens (7.44B unique tokens)
 └─ pretraining/                      DCLM 411m_1x / 1b_1x_fast pretraining, 22-task Core evaluation (eval_sheet.py)

baselines/     rule stacks (C4 / RefinedWeb / FineWeb rules), scrapers (resiliparse, trafilatura, jusText, Dripper),
               ProX-C, UltraX
ablations/     operation ablation (Table 3) and the two-stage "Dripper <extract>" refiner
evaluation/    5,000-page held-out set, page-aligned outputs of every pipeline, teacher fidelity, extraction F1,
               gpt-oss-120b judges, DataMan / FineWeb-Edu scoring, n-gram diversity
analysis/      scripts behind every figure and table
prompts/       every prompt, verbatim
lib/           shared modules (renderer, line operations, executor, rephraser helpers)
```

## Setup

1. Environments (`requirements/README.md`):
   - refiner: Python 3.12, torch 2.9.0+cu128, transformers 4.57.6, vLLM 0.11.1, DeepSpeed, flash-attn 2.8.3, and
     `dripper==1.0.0` (MinerU-HTML, which provides the page renderer) -> `requirements/refiner.txt`;
   - teacher: an environment that can serve Qwen3.8-27B (vLLM 0.28.0, transformers 5.16.1) -> `requirements/teacher.txt`;
   - DCLM: upstream DCLM at commit `c59dd5878c8bd1965b4894e20d8a091e8fdeac7a` plus our patch -> `pretraining/README.md`,
     `requirements/dclm.txt`;
   - tools of the baselines and evaluation (datatrove 0.2.0, trafilatura 2.0.0, jusText 3.0.2, resiliparse 0.16.0,
     gpt-oss-120b via vLLM) -> READMEs of `baselines/` and `evaluation/`.
2. `cp configs/paths.env.example configs/paths.env`, edit, `source configs/paths.env`. All scripts read locations
   from these variables (`RESCRAPER_ROOT`, `WORK_DIR`, `DCLM_DIR`, model identifiers, python interpreters).
3. SLURM scripts are examples written for 8-GPU (H200) nodes. `#SBATCH --partition=gpu|cpu` are placeholders:
   override with `-p` or `SBATCH_PARTITION`; chains use `GPU_PARTITION` / `CPU_PARTITION`.

The source pool is a 10% shard sample of the DCLM `dclm-pool-400m-1x` pool (raw HTML, English pages); every pipeline
starts from the same pages. Held-out shards are listed in `$HELDOUT_SHARDS`.

## Reproducing the released model and corpus

```bash
source configs/paths.env
# teacher 1: Dripper over the pool                               -> data_construction/dripper/README.md
python data_construction/seed_pages/sample_seed_pages.py 1600000
sbatch data_construction/teacher_refine/label_qwen27b.sbatch                       # teacher 2
python data_construction/base_set/select_pages.py $SFT_DIR/base_set/half1          # (+ half2 with PREV_WANTED)
sbatch data_construction/base_set/join_render.sbatch $SFT_DIR/base_set/half1
sbatch data_construction/base_set/build_base_set.sbatch $SFT_DIR/base_set/base_set_norw.jsonl \
       $SFT_DIR/base_set/half1 $SFT_DIR/base_set/half2
# two-stage refiner over the pool (pool-scale deletion proxy)    -> ablations/two_stage_refiner/
sbatch --array=0-13 --export=ALL,WORLD=112 data_construction/rescue/rescue_build.sbatch
sbatch data_construction/rescue/score_edu_pool.sbatch
sbatch --array=0-3 --export=ALL,WORLD=28 data_construction/rescue/rewrite_pool.sbatch  # teacher 3
sbatch training/stage1_chain.sbatch
sbatch training/stage2_chain.sbatch                                                # -> $STAGE2_CKPT
sbatch --array=0-27 --export=ALL,WORLD=224,INFER_MODEL_PATH=$STAGE2_CKPT inference/infer_pool.sbatch
sbatch --dependency=afterany:<array> corpus/finish_chain.sbatch rescraper $STAGE2_CKPT \
       $RESCRAPER_ROOT/prompts/student_system_stage2.txt                            # post-filter, dedup, tokenize
sbatch -J pt1b_rescraper pretraining/pretrain_1b.sh rescraper_norule_noft 29717
sbatch --dependency=afterok:<pt> pretraining/eval_packed.sbatch rescraper_norule_noft 1b
```

Each directory's README lists inputs, outputs, compute and the exact settings of our runs.

## Paper figures and tables

Full map with data files and upstream producers: `analysis/README.md`.

| paper item | produced by |
|---|---|
| Fig. 1(a) rule drops judged worth keeping | `evaluation/judges/` (gpt-oss-120b keep-or-drop judge, per-rule flags) -> `analysis/analyze_rule_groups.py`, `plot_rule_worth_keeping.py` |
| Fig. 1(b) keep-drop accuracy vs. 1B Core | `evaluation/judges/rule_motivation.py` (on `evaluation/heldout_pipeline/` outputs) -> `analysis/plot_keepdrop_vs_core.py` |
| Fig. 2 method overview | `analysis/figure_sources/method_overview.html` -> `analysis/export_html_figure.sh` |
| Table 2 main results (Core, #unique tokens) | `inference/` + `corpus/` (ReScraper), `baselines/` (rule stacks, ProX-C, UltraX), `pretraining/` (`pretrain_400m.sh`, `pretrain_1b.sh`, `core_sheet.sh` / `eval_sheet.py`) |
| Fig. 3 scraper comparison | `baselines/scrapers/` + `baselines/rule_based/`, `baselines/ultrax/`, `pretraining/` -> `analysis/plot_scraper_comparison.py` |
| Table 3 operation ablation | `ablations/operation_ablation/`, `ablations/two_stage_refiner/`, `corpus/`, `pretraining/` |
| Fig. 4 quality before/after each operation | `evaluation/heldout_pipeline/` (DataMan, FineWeb-Edu) -> `analysis/plot_operation_scores.py` |
| Fig. 5 operation mix of model-based pipelines | `evaluation/heldout_pipeline/` -> `analysis/plot_operation_mix.py` |
| Fig. 6 quality by bucket and n-gram diversity | `evaluation/heldout_pipeline/`, `evaluation/diversity/` -> `analysis/plot_quality_diversity.py` |
| Fig. 7 teacher vs. student operations | `evaluation/heldout_set/`, `evaluation/heldout_pipeline/` -> `analysis/plot_decision_sankey.py` |
| Fig. 8 extraction F1 and extraction judge | `evaluation/extraction/` -> `analysis/plot_extraction.py` |
| Table 6 SFT composition | `data_construction/sft_sets/` -> `analysis/count_sft_tags.py`, `analyze_sft_operations.py` |
| Table 7 GPU hours | `analysis/slurm_gpu_hours.py`, `prorate_gpu_hours.py` |
| Table 8 fidelity by input length | `analysis/analyze_fidelity_by_length_5k.py` |
| Fig. 9 token waterfall | `analysis/compute_token_waterfall_5k.py`, `plot_token_waterfall_5k.py` |
| Table 9 per-stage retention | `inference/postfilter.py` (POSTFILTER_STATS), `analysis/stage_retention_from_logs.py`, `token_retention_sample.py` |
| Fig. 10 document length | `evaluation/heldout_pipeline/` -> `analysis/plot_length_distribution.py` |
| Table 11 operation statistics | `analysis/analyze_operation_stats_5k.py` |
| Appendix prompts | `prompts/` |

## Not included

- Data, checkpoints and logs (to be released upon acceptance), and third-party code: DCLM (upstream commit above plus
  `pretraining/dclm_patches/`), Dripper / MinerU-HTML, UltraX, ProX, datatrove, open_lm.
- The launcher/config of the 3B pretraining setting (hyper-parameters in the paper's Table `tab:config`).
- Full-pool runs of some baselines: C4-rule and FineWeb-rule (only page-level implementations of the same rule
  stacks are included), the Raw-data row, trafilatura / jusText over the whole pool (page-level runner included),
  and DataOrchestra (produced with the authors' released system).
- The exact tagging script and training launcher of the two-stage refiner's 45,610-row SFT set (plain applications of
  `data_construction/teacher_refine/` and `training/sft_train.py`), the builder of the intermediate file through which
  the first half of the base set passed (see `data_construction/README.md`), and the builder of the earlier
  3,161-page staged held-out table that forms rows 0-3,160 of the held-out set (`evaluation/heldout_set/README.md`).
- See the "Not included" / "Notes" sections of the directory READMEs for smaller gaps.

## License

Apache-2.0 (`LICENSE`). Third-party models and tools keep their own licenses.
