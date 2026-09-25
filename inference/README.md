# inference/

Pool-scale refinement with the released model (paper Section 3.1 and Appendix A "Pool-scale refinement").

| file | role |
|---|---|
| `infer_pool.py` | per rank: read Dripper step-1 records (raw HTML, `main_html` non-empty), render with the training renderer, number lines, generate with vLLM, execute the program (`lib/rescraper_ops.body_from_prediction_dfirst`), write `{"text", "e2e_tag"}` rows per shard |
| `infer_pool.sbatch` | node-packed array launcher (8 ranks per node, `WORLD = 8 x nodes`), resumable per shard; `INFER_STEMS_FILE` backfills specific shards |
| `postfilter.py`, `postfilter.sbatch` | drop malformed generations (`text/` -> `text_clean/`), with a shard-count gate |

The deterministic executor lives in `lib/rescraper_ops.py`: `<keep>` returns the rendered page minus the `<extract>` lines,
`<edit>` additionally applies its `rm`/`sub` operations (an `<edit>` payload is read as operations only if every line
parses as one), `<rewrite>` returns its payload, `<delete>` emits nothing. Provenance tags written to the corpus:
`<extract>` (keep), `<refine>` (edit), `<rewrite>`.

## Settings of the released corpus

| setting | value |
|---|---|
| model / prompt | `$STAGE2_CKPT`, `prompts/student_system_stage2.txt` |
| decoding | temperature 1.0, top-p 1.0, max 3,072 new tokens, thinking disabled (vLLM 0.11.1, bf16, prefix caching) |
| context | 32,768 tokens; pages whose prompt exceeds 29,696 tokens are skipped |
| omitted pages | empty `main_html`, rendering empty / under 20 characters / over 60 s, prompt too long |
| parallelism | 28 nodes x 8 H200 (WORLD 224), 1.0-1.7 h per node task; 493.9 GPU-h in total including requeues |
| shards | 10,318 of the 10,320 pool shards (two shards contain a page whose rendering never terminates) |
| post-filter | drop documents with >= 2 operation-like lines or a short bare-tag prefix (A2), a tag outside {extract, refine, rewrite} (A3), or a standalone tag line (A5); 6,537 of 10,646,236 documents (0.061%) |

```bash
source configs/paths.env
sbatch --array=0-27 \
  --export=ALL,WORLD=224,INFER_MODEL_PATH=$STAGE2_CKPT,INFER_OUT_DIR=$WORK_DIR/dclm_pipeline/rescraper_corpus/text \
  inference/infer_pool.sbatch
sbatch --dependency=afterany:<array id> corpus/finish_chain.sbatch rescraper $STAGE2_CKPT $RESCRAPER_ROOT/prompts/student_system_stage2.txt
```

`corpus/finish_chain.sbatch` backfills missing shards once, then chains the post-filter, BFF dedup and tokenization.

Note on the released corpus: a small backfill (about 40 of the 10,318 shards) was decoded greedily (T=0) with an
otherwise identical copy of this script; all other shards use the settings above. The operation-ablation pass
(`ablations/operation_ablation/`) re-decodes all 10,318 shards at T=1 and is the control for Table 3.
