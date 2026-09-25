# corpus/

Shared finishing steps for every corpus in the paper (the released ReScraper corpus, the operation-ablation arms, the
two-stage refiner and the model-based baselines): BFF deduplication and GPT-NeoX-20B tokenization with DCLM's tools.
No rule filter and no fastText quality filter is applied ("norule_noft").

| file | role |
|---|---|
| `corpora.sh` | `corpus_vars <name>`: directory table (input of dedup, outputs, DCLM readable name `READ`) for each corpus name |
| `finish_chain.sbatch` | after pool inference: one backfill pass for missing shards, then post-filter -> dedup -> tokenize as a dependency chain |
| `dedup_bff.sh` | DCLM BFF Bloom-filter dedup: 13-grams, threshold 0.8, `--remove-type old-both`, 65e9 expected n-grams, fp-rate 0.01 (not resumable; clears its output first) |
| `tokenize.sh` | DCLM `ray_processing/tokenize_shuffle.py`, `EleutherAI/gpt-neox-20b`, `--content_key text`; writes `$DCLM_DIR/exp_data/datasets/tokenized/<READ>.json` whose `num_tokens` is the "#Unique Tokens" column of the paper tables (must run with cwd = `$DCLM_DIR`) |
| `tokenize_parallel.sh`, `tokenize_merge.py` | optional 8-way parallel tokenization + merge into the same layout (used for the ablation corpora) |

The `bff` binary is built from `$DCLM_DIR/dedup/bff` (`cargo build --release`). Compute: dedup 32 CPUs / 300 GB for
< 1 h; serial tokenization 64 CPUs / 360 GB for ~1.5 h (8 parallel parts: ~11 min each).

```bash
source configs/paths.env
sbatch corpus/dedup_bff.sh rescraper
sbatch --dependency=afterok:<dedup id> corpus/tokenize.sh rescraper
```

Released corpus: `rescraper_norule_noft`, 7,436,788,128 GPT-NeoX-20B tokens in 444 webdataset shards (to be
released upon acceptance as `<HF_ORG>/<DATASET>`). Deduplication removes 6.2% of documents and 9.93% of tokens.
