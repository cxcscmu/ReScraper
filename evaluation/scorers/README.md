# evaluation/scorers/

Document-quality scorers used by the held-out analyses (operation scores, quality buckets; Figures 4 and 6 left).

| file | what |
|---|---|
| `score_dataman.py` | DataMan-1.5B-EN (`RuPeng/DataMan-1.5B-EN`, Peng et al., 2025), `all_rating` prompt, greedy (T=0, seed 1024), 64 output tokens; the 14th rating line is the overall score (1-5; anything else is stored as invalid). `--inputs name=path.jsonl ... --output-dir OUT` -> `OUT/<name>.scores.jsonl` (one line per non-empty input text, in order) + `summary.json/.tsv`. Empty / whitespace-only texts are skipped silently, so callers send only non-empty texts. |
| `simple_dataman.py` | DataMan inference wrapper (vLLM): the model's English request templates, ellipsis / image-link regex filtering, truncation to 20,000 characters (head + tail) and 1,894 tokens (head + "..." + tail), chat template, output parsing. |
| `score_edu.py` | FineWeb-Edu classifier (`HuggingFaceFW/fineweb-edu-classifier`), raw regression output, 512-token truncation, batch 64, GPU if available. `--inputs name=path.jsonl ... --output-dir OUT` -> `OUT/<name>.edu.jsonl` (`{"edu": score}` per line, rounded to 4 decimals) + `edu_summary.json`. |

Both are called once per 10,000-text shard by `heldout_pipeline/stages/s3_scores.sbatch` (1 GPU; DataMan dominates the
time, about 20 min including model loads for the ~24k unique texts of the greedy run on one 80 GB GPU). Environment: `REFINER_PY` (vLLM 0.11.1, transformers,
torch; the DataMan wrapper also imports `openai` for its optional server mode), `HF_HOME`.
