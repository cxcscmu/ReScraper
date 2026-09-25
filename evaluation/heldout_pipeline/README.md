# evaluation/heldout_pipeline/

Page-aligned cross-system pipeline for Section 5: every system processes the same held-out pages
(`heldout5k.jsonl`, `evaluation/heldout_set/`) and every per-page output is keyed by `gid`.

Two runs of the released model exist, and the paper's figures use the second:
- **greedy** (`run_all.sh`, work dir `work/heldout5k`): the released checkpoint run on the page table with T=0 and up to
  8,192 new tokens (`stages/s1_ours_infer.py`), plus all four baselines and all scores. Kept as the reference run; the
  release run reuses its baseline outputs, its DataMan/Edu scores (by exact text) and its prompt/target token counts.
- **release decoding** (`run_release.sh`, work dir `work/heldout5k_rel`, figure data `data/*_5k_rel.json`): the
  decoding of the released corpus (T=1.0, top-p 1.0, up to 3,072 tokens, `inference/infer_pool.py`). Its per-page records
  come from the full arm of the operation ablation (`ablations/`, `$ABLATION_DIR/raw`), which ran the same model, prompt,
  decoding and executor over the pool; the held-out pages are joined on (stem, Dripper step-1 index) with the HTML md5
  asserted. 11 of 5,000 pages exceed the context and were skipped by that run.

## Stages
| job | script | what | code it follows | resources (5,000 pages) |
|---|---|---|---|---|
| A | `stages/s1_ours_infer.py` (+ `.sbatch`) | released student (`$STAGE2_CKPT`, system prompt `prompts/student_system_stage2.txt`) on `input`: bf16, greedy, max_model_len 32768, max_tokens min(8192, 32768 - plen); pages with plen + 256 > 32768 are not sent | held-out evaluation of the SFT runs (decision-first reader) | 1x 48 GB GPU, ~20 min |
| B | `stages/s2_proxc.py` | ProX-C on `resiliparse` | the pool ProX-C run (`baselines/prox_c/`): chunking, plain prompt, T=0 / top-p 0.9 / 256 tokens, max_len 2048, executor thresholds 0.0 / 0.95, error_op 2 (`lib/prox_chunk_utils.py`) | 1 GPU, ~5 min |
| C | `stages/s2_rw_rule.py` | RefinedWeb-rule | `lib/rw_rule_chain.py` = DCLM mappers of `dclm_baseline_refinedweb_post_lang.yaml`, page by page (`DCLM_PY`, `DCLM_DIR`) | CPU |
|   | `stages/s2_fw_rule.py` | FineWeb-rule | `lib/fw_rule_chain.py` = the FineWeb-rule baseline's datatrove 0.2.0 chain (URL filter, blank-line strip, lang 0.65, Gopher repetition/quality, C4 without the terminal-punctuation filter, FineWeb quality; `DATATROVE_PY`) | CPU |
|   | `stages/s2_ultrax.sh` -> `s2_ultrax_finish.py keys`, `s2_ultrax_join.py`, `s2_ultrax_finish.py finish` | UltraX per page | the UltraX run over the raw resiliparse pool (`baselines/ultrax/`, `$ULTRAX_RERUN_DIR/{post,inf}`), joined by exact `original` text within the stem | CPU, 1 min |
| D | `stages/s1_ours_exec.py`, `stages/s3_gather.py` | execute the student programs (`lib/ours_exec.py`), teacher fidelity (`evaluation/fidelity/teacher_metrics.py`), collect every text to score | `lib/e2e_ops.py` executor | CPU, 2-3 min |
| E | `stages/s3_scores.sbatch` | DataMan + FineWeb-Edu on every unique text | `evaluation/scorers/score_dataman.py`, `score_edu.py`, one call per 10k-text shard | 1 GPU, ~20 min for ~24k texts |
| F | `stages/s4_build.py` | figure data files | see Outputs | CPU, ~1 min |
| release | `stages/rel_chain.sbatch`: `s1_rel_extract.py` -> `s1_rel_exec.py` -> `s3_gather.py` -> `s3_reuse.py`; `s3_scores.sbatch`; `rel_build.sbatch` (`s4_build.py` with `HP_DECODING=release`) | release-decoding records, fidelity and decision flow, score reuse, figure data | | CPU + 1 GPU for the ~2k new texts, ~15 min |

`run_all.sh <page_table>` submits A-F with dependencies; every stage skips work whose output exists (delete a stage's
outputs in `work/<tag>/` to force it), and a page-table md5 guard refuses to mix outputs of different tables.
`run_release.sh <page_table>` then submits the release chain. The first run of `s2_fw_rule.py` downloads datatrove's
URL block lists (needs network or a warm HF cache).

Scoring: texts are deduplicated by exact string before scoring. Only non-empty strings are sent (the filter
`score_dataman.py` applies), so scorer output lines map 1:1 to the unique-text list; `s4_build.py` asserts the counts
per shard and re-attaches scores to gids through `work/<tag>/texts_index.jsonl`. DataMan outputs outside 1..5 are stored
as null.

## Outputs (`$HELDOUT_PIPELINE_DIR/data`, suffix `_5k` greedy / `_5k_rel` release)
| file | definition |
|---|---|
| `decision_flow_5k_rel.json` | teacher tag x student decision on all 5,000 pages; skipped / unparseable pages count as delete (`s1_rel_exec.py`) |
| `operation_scores_<suf>.json` | DataMan and FineWeb-Edu per page grouped by the student's decision (line 1 of the raw program); before = `apply_ops(input, stage-1 ops).strip()`, after = executed text `.strip()`; keep and delete keep only before; empty texts dropped; `gid` list per group |
| `operation_mix_<suf>.json` (+ `operation_mix_perpage_<suf>.jsonl`) | per system (ReScraper, UltraX, ProX-C): keep / edit removing < 10%, 10-50%, > 50% of words / delete / rewrite / not_processed (ReScraper: prompt too long or unparseable decision; UltraX / ProX-C: no resiliparse text or not in the run) |
| `quality_buckets_<suf>.json` | pre = DataMan / Edu of `resiliparse`; post = of each system's output, null if deleted; order refinedweb_rule, fineweb_rule, proxc, ultrax, rescraper |
| `dist_length_<suf>.json` | GPT-NeoX-20B token and character length of every kept output, with `gids`; fineweb_rule and resiliparse included |
| `scores_by_gid_<suf>.jsonl`, `build_report_<suf>.json` | per-page scores and status of every system; counts and checks |

Per-system per-page outputs (`work/<tag>/`): `ours_raw.jsonl` (raw program, plen, tlen, finish_reason; greedy run),
`ours.jsonl` (status, decision, pt, final text, pre-op extraction, operation class, emits; the release run adds `raw`,
`in_clean`, `f3_dropped`), `ours_validation.json` (fidelity vs the teacher, rescue branch, length caps and loops),
`sys_{ultrax,proxc,refinedweb_rule,fineweb_rule}.jsonl` (`{gid, text|null, status, ...}`), `ux_out.jsonl`.

## Numbers of our runs (for orientation)
Release decoding, 4,989 pages sent: decision accuracy 74.80 (95% CI 73.6-76.0), keep-or-delete 89.32, token
P / R / F1 89.11 / 89.41 / 89.26 (CI 88.1-90.3). Re-executing each ablation program on the table's `input` reproduces
the ablation tag, final and extracted text on 4,989/4,989 pages. Kept (non-empty) documents: RefinedWeb-rule 2,963,
FineWeb-rule 2,155, ProX-C 4,318, UltraX 4,287, ReScraper 3,538. On the first 960 pages the RefinedWeb-rule texts equal the pool's own
RefinedWeb-rule output for the same records (960/960 membership, 507/507 text).

## Environment
`WORK_DIR`, `RESCRAPER_ROOT`, `HF_HOME`, `STAGE2_CKPT`, `DCLM_DIR`, `NLTK_DATA` (nltk data used by the DCLM mappers and
the datatrove filters); interpreters `REFINER_PY` (vLLM 0.11.1, also needs
pyarrow for the UltraX join), `DCLM_PY`, `DATATROVE_PY`; inputs `ULTRAX_RERUN_DIR`, `ABLATION_DIR` (and optionally
`F3_DROP_LOG`, the log of the ablation build whose `F3_DROP <stem>:<idx>` lines mark the malformed `<extract>` generations
it dropped); outputs `HELDOUT_PIPELINE_DIR` (default `$WORK_DIR/eval/heldout_pipeline`); overrides `HP_WORK`, `HP_OUT`,
`HP_SUF`, `HP_DECODING`, `OURS_CHUNK`, `PROXC_CHUNK`, `PROXC_MODEL`, `SCORE_SHARD`. SLURM partitions are placeholders
(`GPU_PARTITION` / `CPU_PARTITION`, or `-p`).
