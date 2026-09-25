# pretraining/

DCLM pretraining and Core evaluation wrappers (paper Section 4, Tables 1 and 3, Appendix Table `tab:config`).

We do not vendor DCLM. Use the upstream repository `mlfoundations/dclm` at commit
`c59dd5878c8bd1965b4894e20d8a091e8fdeac7a` and apply our modifications:

```bash
git clone https://github.com/mlfoundations/dclm $DCLM_DIR && cd $DCLM_DIR
git checkout c59dd5878c8bd1965b4894e20d8a091e8fdeac7a
git apply $RESCRAPER_ROOT/pretraining/dclm_patches/dclm_modifications.patch
cp -r $RESCRAPER_ROOT/pretraining/dclm_patches/eval/shards eval/shards
cp $RESCRAPER_ROOT/pretraining/dclm_patches/baselines_configs/*.yaml baselines/baselines_configs/
(cd dedup/bff && cargo build --release)
```

`dclm_patches/dclm_modifications.patch` contains only what our runs need:
- `training/params.py`: `--ignore-parse-errors` / `--dataset-resampled` pass-through to open_lm (we use
  `--ignore-parse-errors`), and open_lm's `--delete-previous-checkpoint` disabled (see the pruner in `pretrain_1b.sh`);
- `ray_processing/tokenize_shuffle.py`, `ray_processing/utils.py`: local (non-S3) output paths and sizes;
- `eval/eval_openlm_ckpt.py`: `makedirs(exist_ok=True)` and the tokenizer taken from the argument;
- `baselines/`: a `resiliparse_extraction_modifier`, the RefinedWeb rule configs split into a language/URL part
  (`dclm_baseline_refinedweb_lang_only.yaml`) and the post-language rules (`dclm_baseline_refinedweb_post_lang.yaml`),
  a stub `baselines/core/monitoring.py`, `ray_processing/process_no_ray.py` (single-node rule processing), and
  `resiliparse` added to DCLM's requirements.
`dclm_patches/eval/shards/chunk0-6.yaml` split DCLM's `eval/mmlu_and_lowvar.yaml` into 7 chunks that run in parallel;
merging them (`eval_merge.py`) reproduces the monolithic evaluation.

| file | role |
|---|---|
| `pretrain_400m.sh` | 400M setting: `--scale 411m_1x` (412M model, 8.2B training tokens), 4 nodes x 8 GPUs |
| `pretrain_1b.sh` | 1B setting: `--scale 1b_1x_fast` (1.4B model, 28.8B training tokens: 54,923 steps x 256 x 2048), 8 nodes x 8 GPUs, ~4.3 h; optional checkpoint pruner (`PRUNE=1`) |
| `eval_chunk.sh` | one eval chunk on 1 GPU (array 0-6) |
| `eval_packed.sbatch` | all 7 chunks on one 8-GPU node (~1.25 h for 1.4B), checks that the evaluated checkpoint is final, then `core_sheet.sh` |
| `eval_merge.py` | merges the 7 chunk jsons into one `metrics_mmlu_and_lowvar.json` (refuses partial merges) |
| `core_sheet.sh`, `eval_sheet.py` | Core = centered accuracy averaged over the 22 low-variance tasks (`commonsense_qa` dropped) plus the per-category averages reported in Tables 1 and 3 |

Common flags: `--num-checkpoints 20 --multiple-data-passes --ignore-parse-errors --data-tolerate-num-ckpts 20
--data-tolerate-error-p 0.5`, seed 124 (DCLM default). Every corpus is repeated as needed to fill the fixed token
budget, so always report its unique token count (`num_tokens` of the tokenized dataset json) next to accuracy.

```bash
source configs/paths.env
sbatch -J pt1b_rescraper pretrain_1b.sh rescraper_norule_noft 29717
sbatch -J eval_rescraper --dependency=afterok:<pt id> eval_packed.sbatch rescraper_norule_noft 1b
# -> $CORE_DIR/core_1b_rescraper_norule_noft_LATEST.tsv
```

Compute: a 1.4B run costs 225-254 H200 GPU-h (paper Table `tab:teacher-cost`).

Not included: the launcher/config of the 3B setting (2.8B model, 55.9B training tokens, 106,604 steps x 256 x 2048,
Table `tab:config`); it used DCLM's `training.train` with the hyper-parameters listed in that table.
