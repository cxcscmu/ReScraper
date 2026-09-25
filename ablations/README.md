# ablations/

## operation_ablation/ (paper Table 3, `tab:ablation-supervision`, 1B setting)

All arms come from ONE new full-pool pass of the released model with the release decoding (T=1.0, top-p 1.0,
3,072 new tokens), which writes the release text rows (`full/text`, the control) and a per-page raw sidecar
(`raw/`: generation, executor tag and text, and the text after the model's own `<extract>` ops). Each arm disables
one operation when converting the model's outputs into text; the model and its generations are identical.

| paper row | arm | rule | script | unique tokens |
|---|---|---|---|---:|
| (control) | `full` | executor as released | `infer_pool_with_raw.py` | 7.43B |
| w/o `<edit>` | `noedit` | `<edit>` pages keep the extracted text (edit removals not applied) | `build_arms.py` | 7.77B |
| w/o `<rewrite>` | `norwdel` | pages the model rewrote are dropped (the rescue is not performed) | `build_arm_norwdel.py` | 6.62B |
| w/o `<delete>` | `nodel` | `<delete>` pages keep the extracted text | `build_arms.py` | 9.08B |
| w/o `<extract>` | `noext` | the program is executed without its `<extract>` block (kept/edited pages keep the whole rendering) | `build_arm_noext.py` | 14.10B |
| (not in the paper) | `norw` | rewritten pages keep the extracted text | `build_arms.py` | 7.63B |
| Dripper `<extract>` | two-stage refiner | Dripper's extracted text, then a text-to-text refiner distilled from the same refining teacher | `../two_stage_refiner/` | 7.34B |

The 40 pages whose `<keep>` program leaves operation residue (F3) are dropped from every arm. The ReScraper row of
Table 3 is the released corpus (7.44B); the `full` arm is its statistical replica from the same model and decoding.

Steps (`source configs/paths.env`; `ABL=$WORK_DIR/dclm_pipeline/operation_ablation`):
1. `sbatch --array=0-27 --export=ALL,WORLD=224,INFER_MODEL_PATH=$STAGE2_CKPT infer_pool_with_raw.sbatch` (28 x 8 H200)
2. `sbatch build_arms.sbatch` (shard gate, builds nodel / noedit / norw, verifies the full arm byte for byte)
3. `sbatch --array=0-15 build_arm_norwdel.sbatch 16 16` and `sbatch --array=0-31 build_arm_noext.sbatch 32 32`
4. for each arm: `inference/postfilter.sbatch ablation_<arm>` (with `EXPECT_SHARDS` from step 2) ->
   `corpus/dedup_bff.sh ablation_<arm>` -> `corpus/tokenize_parallel.sh` + `tokenize_merge.py` ->
   `pretraining/pretrain_1b.sh ablation_<arm>_norule_noft` (8 nodes) -> `pretraining/eval_packed.sbatch ... 1b`

## two_stage_refiner/ (paper Table 3 row "Dripper `<extract>`"; also used in data construction)

A Qwen3-0.6B text-to-text refiner: input = Dripper's extracted text, output = `<extract>` (keep as is) |
`<refine>` + refined text | `<delete>`, distilled from the same Qwen3.8-27B teacher and prompt
(`prompts/teacher_refine_qwen27b.txt`) on 45,610 pages of a separate Dripper SFT source; system prompt
`prompts/two_stage_refiner_system.txt`; tag-loss weight 5; 5 epochs. Its pool output (`two_stage_pool/best`) is the
Dripper `<extract>` corpus of Table 3 and is also the pool-scale proxy for the teacher's deletions used by
`data_construction/rescue/`.

| file | role |
|---|---|
| `build_t2t_set.py` | joins the tagged teacher labels (keyed by page HTML) to the Dripper text of the same pages |
| `infer_t2t.py`, `infer_t2t.sbatch` | pool inference (greedy, 100k-character chunks), `two_stage_pool/input` -> `two_stage_pool/best` |

Teacher labels for its 45,610 pages come from `data_construction/teacher_refine/label_qwen27b.py` run on that source
(`T27_SRC`, rows keyed by line index) with the same tag rule as `data_construction/base_set/select_pages.py`.
Training uses `training/sft_train.py` with the arguments above. `two_stage_pool/input` is a link to
`$DRIPPER_STEP2_DIR`. Not included: the exact tagging script of the 45,610-row set and its training launcher
(both were plain applications of the tools named above).
