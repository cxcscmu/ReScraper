# evaluation/

Everything behind the Section-5 analyses and Figure 1: the 5,000-page held-out set, the page-aligned run of ReScraper
and the baselines on it, teacher fidelity, extraction quality, the LLM judges, the quality scorers and the corpus-level
n-gram diversity. All analyses use one held-out table, `heldout5k.jsonl` (5,000 pages from 5 pool shards that no
training stage saw; its first 960 rows are `heldout960`), labelled by the teacher cascade with the Stage-2 rewrite rule
(FineWeb-Edu >= 1.0), processed by every pipeline and keyed by `gid`.

| dir | what | paper item |
|---|---|---|
| `heldout_set/` | builds `heldout5k.jsonl` (+ `html5k.jsonl`): shard exclusion by stem, rendering, Dripper join, Qwen3.8-27B labels, FineWeb-Edu, RePro rewrites, edu1 relabel, resiliparse join | all of Section 5, Figure 1 |
| `heldout_pipeline/` | ReScraper (greedy run and release-decoding run), UltraX, ProX-C, RefinedWeb-rule, FineWeb-rule on every held-out page; DataMan / FineWeb-Edu scores; figure data | Figures 4-7, 10; inputs of Tables 8, 11 and Figure 9 |
| `fidelity/` | teacher-fidelity scoring (decision accuracy, token P/R/F1 with deleted pages -> "", decision flow) | Figure 7, Table 8 |
| `extraction/` | extraction token/line F1 vs Dripper; gpt-oss-120b extraction judge | Figure 8 |
| `judges/` | gpt-oss-120b keep-or-drop judge; independent per-rule flags of the two rule stacks | Figure 1 |
| `scorers/` | DataMan and FineWeb-Edu scorers | Figures 4, 6 |
| `diversity/` | distinct n-grams of each pipeline's output on the same 376 pool shards | Figure 6 (right) |

Judge prompts: `prompts/judge_extraction.txt`, `prompts/judge_keep_or_drop.txt` (the exact files the runs used; printed
verbatim in the appendix).

## Run order
```bash
source configs/paths.env                                   # WORK_DIR, RESCRAPER_ROOT, HF_HOME, interpreters, ...
bash evaluation/heldout_set/run_all.sh                     # -> $WORK_DIR/eval/heldout5k/{heldout5k,html5k}.jsonl
PT=$WORK_DIR/eval/heldout5k/heldout5k.jsonl
bash evaluation/heldout_pipeline/run_all.sh $PT            # greedy run + all baselines + scores
bash evaluation/heldout_pipeline/run_release.sh $PT        # release decoding (needs ablations/operation_ablation raw rows)
bash evaluation/extraction/run_all.sh $PT $WORK_DIR/eval/heldout_pipeline/work/heldout5k_rel/ours.jsonl \
     $WORK_DIR/eval/extraction/runs/heldout5k_rel
sbatch evaluation/judges/keep_judge.sbatch $WORK_DIR/eval/extraction/runs/heldout5k_rel/ext_texts_5000.jsonl \
     $WORK_DIR/eval/keep_judge/keep_judge_5000.jsonl
sbatch evaluation/judges/allrules/run.sbatch $PT $WORK_DIR/eval/heldout_pipeline/work/heldout5k_rel \
     $WORK_DIR/eval/keep_judge/keep_judge_5000.jsonl $WORK_DIR/eval/keep_judge
sbatch evaluation/diversity/fw.sbatch; sbatch evaluation/diversity/count.sbatch rescraper ultrax proxc refinedweb_rule fineweb_rule resiliparse_raw
python evaluation/diversity/build_json.py $FIG_DATA_DIR/diversity_ngrams_pool376.json   # FIG_DATA_DIR: see analysis/README.md
```
Upstream inputs: the Dripper pool run (`data_construction/dripper/`), the two-stage pool run and rescue sidecars
(`ablations/two_stage_refiner/`, `data_construction/rescue/`), the baselines over the pool (`baselines/`), the released
corpus (`inference/`) and the operation ablation's full arm (`ablations/operation_ablation/`).

## Greedy vs release decoding
The released model was run twice on the held-out table. The **release-decoding run** (T=1.0, top-p 1.0, 3,072 new
tokens; the decoding of the released corpus; work dir `heldout_pipeline/work/heldout5k_rel`, figure data
`*_5k_rel.json`) is behind **every** paper figure and table of Section 5 and the appendix, and behind Figure 1. The
greedy run (`work/heldout5k`, `*_5k.json`) is the reference run whose baseline outputs, scores and token counts the
release run reuses. (The description string inside the paper's `operation_scores.json` still says "greedy run"; the file
itself is the release-decoding output.)

## Pipeline output -> figure data file in `analysis/data`
Figure data files are written under `$WORK_DIR/eval/` and copied to `analysis/data` (or `$FIG_DATA_DIR`) under the names
the plotting scripts read.

| output of this directory | name in `analysis/data` | used by | run |
|---|---|---|---|
| `heldout_pipeline/data` `decision_flow_5k_rel.json` (`stages/s1_rel_exec.py`; also `fidelity/score_fidelity.py`) | `decision_flow.json` | Fig. 7 (`plot_decision_sankey.py`) | release |
| `heldout_pipeline/data` `operation_scores_5k_rel.json` (`stages/s4_build.py`) | `operation_scores.json` | Fig. 4 (`plot_operation_scores.py`) | release |
| `heldout_pipeline/data` `operation_mix_5k_rel.json` (`stages/s4_build.py`) | `operation_mix.json` | Fig. 5 (`plot_operation_mix.py`) | release; the paper file additionally holds a `DataOrchestra` entry with counts computed separately from the DataOrchestra run on the same 5,000 pages (not produced here) |
| `heldout_pipeline/data` `quality_buckets_5k_rel.json` (`stages/s4_build.py`) | `quality_buckets.json` | Fig. 6 left (`plot_quality_diversity.py`) | release |
| `heldout_pipeline/data` `dist_length_5k_rel.json` (`stages/s4_build.py`) | `dist_length.json` | Fig. 10 (`plot_length_distribution.py`) | release |
| `diversity/build_json.py` output (from `$DIV_DIR/results/*.json`) | `diversity_ngrams_pool376.json` | Fig. 6 right (`plot_quality_diversity.py`) | corpora on 376 pool shards |
| `extraction/runs/<run>/extraction_quality.json` (`make_extraction_quality.py` = `ext_vs_dripper_5000.json` + `judge/ext_judge_5000.json`) | `extraction_quality.json` | Fig. 8 (`plot_extraction.py`) | release (student extraction) |
| `heldout5k/heldout5k.jsonl` + `heldout_pipeline/work/heldout5k_rel/ours.jsonl` | inputs of `analyze_fidelity_by_length_5k.py` -> `fidelity_by_length_5k.json`; `analyze_operation_stats_5k.py` -> `operation_stats_5k.json`; `compute_token_waterfall_5k.py` -> `token_waterfall_5k.json` | Table 8, Table 11, Fig. 9 | release |
| `judges/`: `keep_judge_5000.jsonl` + `heldout_pipeline/work/heldout5k_rel/{ours,sys_*}.jsonl` | inputs of `judges/rule_motivation.py` (values in `analysis/plot_keepdrop_vs_core.py`) | Fig. 1(b) | release |
| `judges/allrules/`: `rule_flags_5000.jsonl` | input of `analyze_rule_groups.py` (values in `plot_rule_worth_keeping.py`) | Fig. 1(a) | release |

Default output locations: `$WORK_DIR/eval/{heldout960,heldout5k,heldout_pipeline,extraction,keep_judge,diversity}`.

## Environment variables
All paths come from the environment (`configs/paths.env.example`); `evaluation/eval_paths.py` (Python) and
`evaluation/env.sh` (SLURM jobs, needs `RESCRAPER_ROOT`) hold the defaults.
- Required: `WORK_DIR`, `RESCRAPER_ROOT`, `HF_HOME`; models `STAGE2_CKPT` (the released ReScraper model), `RW_MODEL` (the
  public RePro 1B rephraser checkpoint (Yu et al., 2025); identifier withheld for anonymity), `TEACHER_MODEL`
  (`Qwen/Qwen3.8-27B`); `DCLM_DIR` (DCLM checkout: RefinedWeb-rule mappers and the fastText LID model, or `LID_MODEL`).
- Interpreters: `REFINER_PY` (vLLM 0.11.1 / transformers 4.57: student, ProX-C, RePro, scorers, rendering),
  `TEACHER_PY` (Qwen3.8-27B), `PIPE_PY` (fasttext), `DCLM_PY` (DCLM mappers, resiliparse 0.16.0), `DATATROVE_PY`
  (datatrove==0.2.0, FineWeb-rule), `JUDGE_PY` (gpt-oss-120b).
- Inputs (defaults under `$WORK_DIR`): `HTML_POOL_DIR`, `POOL_SIMP_DIR`, `DRIPPER_STEP1_DIR`, `DRIPPER_STEP2_DIR`,
  `RESILIPARSE_EXTRACT_DIR`, `TWO_STAGE_POOL_DIR`, `TEXT_DECISIONS_DIR`, `ULTRAX_RERUN_DIR`, `ABLATION_DIR`
  (+ optional `F3_DROP_LOG`), `SFT_DIR`, `SFT_DATA_DIR`, `HELDOUT3161_DIR`, `STAGE1_SFT_STEMS`,
  `TWO_STAGE_SFT_SOURCE_GLOB`, `RESCRAPER_CORPUS_DIR`, `PROXC_POOL_DIR`, `RW_RULE_POOL_DIR`.
- Outputs: `EVAL_DIR` (default `$WORK_DIR/eval`), `HELDOUT960_DIR`, `HELDOUT5K_DIR`, `HELDOUT_PIPELINE_DIR`, `DIV_DIR`.
- SLURM: `#SBATCH --partition=gpu|cpu` are placeholders; the drivers pass `-p ${GPU_PARTITION:-gpu}` /
  `-p ${CPU_PARTITION:-cpu}`, and `SBATCH_PARTITION` works for direct submissions. Logs go to `logs/` under the
  submission directory.

## Compute (our runs)
| step | hardware | wall time |
|---|---|---|
| held-out set: 960 relabel | 1 GPU (48 GB) | minutes |
| held-out set: CPU steps 1, 1b, 2, 4 | 8-16 CPUs | ~20 + 10 + 15 + 5 min |
| held-out set: Qwen3.8-27B labels + RePro rewrites | 1 node, 8x H200 | ~10 min |
| held-out pipeline: student (greedy), ProX-C | 1 GPU each (48 GB) | ~20 min, ~5 min |
| held-out pipeline: rule stacks + UltraX join | 4 CPUs | ~5 min |
| held-out pipeline: DataMan + FineWeb-Edu | 1 GPU (80 GB) | ~20 min |
| extraction judge (25,250 requests) / keep-or-drop judge (5,000) | 1x 96 GB GPU (RTX PRO 6000 Blackwell) per task | 2 x 45 min / 7 min |
| rule flags | 4 CPUs | < 1 h |
| n-gram diversity, 376 shards, 6 corpora | 24 CPUs, 400 GB | 7-22 min per corpus |
