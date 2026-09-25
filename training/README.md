# training/

Two-stage supervised fine-tuning of Qwen3-0.6B on the serialized targets (paper Section 3.3, Appendix A).

| file | role |
|---|---|
| `sft_train.py` | HF Trainer + DeepSpeed ZeRO-3 SFT: packed 32,768-token sequences, loss on the answer only, chunked cross-entropy, `--tag_loss_weight` on the leading decision-tag tokens, `--loss_scale 24` (see below), `--prepacked_dir` to load a prepacked dataset |
| `deepspeed_zero3.json` | DeepSpeed config (bf16, ZeRO-3, gradient clipping 1.0) |
| `prepack_sft.py`, `prepack_shard.py`, `prepack_merge.py`, `prepack_light_check.py` | tokenize + pack an SFT jsonl once on CPU (8 shards merged), identical preprocessing to `sft_train.py` |
| `prepack_shard.sbatch`, `prepack_merge.sbatch` | SLURM wrappers of the prepack |
| `train_multinode.sbatch` | srun + torchrun launcher; enforces global batch 192 |
| `train_wrap.sbatch` | computes `save_steps` = one epoch from the prepack and launches `train_multinode.sbatch` |
| `stage1_chain.sbatch` | Stage 1: `build_stage1_set.py` -> `finalize_prompt.py` -> prepack -> train |
| `stage2_chain.sbatch` | Stage 2: `build_stage2_set.py` (scheme `edu1`, rewrite 30%) -> prompt -> prepack -> continue training from the Stage-1 epoch-2 checkpoint |

## Settings used for the released model

| | Stage 1 | Stage 2 |
|---|---|---|
| init | `Qwen/Qwen3-0.6B` | Stage-1 `checkpoint-2244` (end of epoch 2 of 3) |
| data | `$SFT_DIR/stage1_set.jsonl`, 1,383,115 rows -> 1,376,248 packed samples (215,362 bins) | `$SFT_DIR/stage2_set.jsonl`, 131,484 rows (21,387 bins) |
| system prompt | `prompts/student_system_stage1.txt` | `prompts/student_system_stage2.txt` |
| epochs / steps | 3 / 3,366 (1,122 per epoch) | 3 / 336 (112 per epoch) |
| topology | 8 nodes x 8 GPUs, micro 1, accum 3 | 4 nodes x 8 GPUs, micro 1, accum 6 |
| global batch | 192 | 192 |
| peak LR / schedule | 8e-5, linear warmup (3% of steps), cosine decay, weight decay 0.01 | same (warmup ~11 steps) |
| max sequence length | 32,768, packed | 32,768, packed |
| precision | bf16 | bf16 |
| decision-tag loss weight | 5 | 5 |
| cost (H200 GPU-h) | 244.4 (up to checkpoint-2244) | 37.3 |

Stage 2 ran in about 70 minutes on 4 nodes of 8 H200 (~12 s/step).

```bash
source configs/paths.env
sbatch training/stage1_chain.sbatch                        # builds the Stage-1 set, prepacks, trains -> $CKPT_DIR/stage1
sbatch training/stage2_chain.sbatch                        # defaults = paper setting -> $CKPT_DIR/stage2 (the released model)
```

**Batch topology.** `train_multinode.sbatch` refuses any configuration whose micro x accum x nodes x 8 is not 192.
With `--loss_scale 24` (default) the accumulated gradient is 24x the per-micro-batch mean, which reproduces the
original single-node regime (micro 1, accum 24) where the gradient is clipped at almost every step; use one of the
calibrated pairs nodes/accum = 1/24, 2/12, 4/6, 8/3 when changing the node count.

The chains are idempotent: they skip the set build, the prompt rendering and the prepack when their outputs exist;
training resumes from the newest checkpoint on requeue. Held-out evaluation of the checkpoints is in `evaluation/`.
