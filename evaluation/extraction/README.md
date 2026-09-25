# evaluation/extraction/

Extraction quality of ReScraper's `<extract>` step (Figure 8): token/line F1 against Dripper, its extraction teacher
(left), and a reference-free gpt-oss-120b extraction judge (right), on the 5,000 held-out pages.

## Inputs
- **Page table** `heldout5k.jsonl` (`evaluation/heldout_set/`): `input` (the `<lid:n>` rendering), `drip` (Dripper text),
  `resiliparse` (pool extraction).
- **HTML table** `html5k.jsonl` `{gid, html}` (picked up next to a 5,000-row page table; `HTML=` overrides).
- **Student outputs**: one row per gid. `lib/common.py:load_student` accepts the `ours.jsonl` of the held-out pipeline
  (`text`, `decision`, `extracted`, `status`) or raw programs (`raw`), from which final text and pre-op extraction are
  derived with `lib/e2e_ops.py`. The paper uses the release-decoding run, `heldout_pipeline/work/heldout5k_rel/ours.jsonl`.
  A row with `status != ok` (11 pages over the context) becomes decision `omitted`, empty extraction.

## A. Extraction vs Dripper (`extract_baselines.py`, `resiliparse_fill.py`, `ext_metrics.py`)
Systems:
- student: the pre-op extraction (`apply_ops(input, <extract> removals)`);
- resiliparse: the page-table `resiliparse` field; for null rows the re-extraction of `resiliparse_fill.py` with the pool
  settings (`extract_plain_text(HTMLTree.parse(html), main_content=True, alt_texts=False, preserve_formatting=True)`,
  resiliparse 0.16.0; byte-identical to the pool text on 960/960 checked pages; 0 null rows in heldout5k);
- trafilatura 2.0.0: `trafilatura.extract(html)`, all defaults;
- jusText 3.0.2: `justext.justext(html, get_stoplist("English"))`, default thresholds, non-boilerplate paragraphs joined
  with "\n".
Metrics (reference = Dripper text): token P/R/F1 with `\w+` on lower-cased text, multiset overlap, empty system -> P=1,
empty reference -> R=1 (the fidelity-by-length convention); line P/R/F1 on multisets of normalised lines (lowercase, runs
of non-word characters -> one space); empty share; macro means with a page bootstrap (B=1000) and pooled values.
Outputs: `ext_vs_dripper_<N>.json`, `.perpage.jsonl`, `ext_texts_<N>.jsonl` (the texts the judges read; also the input of
`evaluation/judges/keep_judge.py`). CPU: seconds for metrics; trafilatura + jusText ~1 CPU-min per 1,000 pages.

## B. Extraction judge (`judge_run.py`, `judge.sbatch`, `agg_extract.py`)
- Judge `openai/gpt-oss-120b`, pointwise, no reference extraction; prompt `prompts/judge_extraction.txt` (the prompt of
  the appendix, verbatim). The judge sees the rendered page (`input` without `<lid:n>`) and ONE extraction and returns
  main_content_recall, boilerplate_precision and integrity (0-2 each) plus a rationale as JSON. The paper reports the mean
  of the three dimensions.
- vLLM 0.11.1, `LLM.chat`, temperature 0, `reasoning_effort=low`, max_tokens 2048, max_model_len 32768, 24,000-character
  truncation per text (`"[... truncated for length]"` appended; the prompt tells the judge not to penalise it), token
  guard, last-valid-JSON parsing, chat-template date pinned (`JUDGE_DATE`) so reruns see identical prompts, resumable
  1,000-request units, gid sharding (`SHARD_N`).
- Systems student, Dripper, resiliparse, trafilatura, jusText, submitted per page in `random.Random(gid)` order (position
  recorded; no order effect).
- Sanity controls, run first on 50 pages (Dripper >= 5 lines and >= 100 words, rendered page >= 1.3x Dripper): empty
  extraction, full rendered page, first 40% of Dripper lines, Dripper text twice, words shuffled per line; the job exits
  3 if fewer than 99% parse. On heldout5k the controls pass 50/50, 50/50, 49/50 and 50/50, except whole-text
  duplication, which is under-penalised (27/50 "twice" controls get integrity <= 1).
- Hardware: 1x 96 GB GPU (RTX PRO 6000 Blackwell) per array task, TP1, `VLLM_MXFP4_USE_MARLIN=1`; ~10 requests/s,
  ~6 min model load; 5,000 pages x 5 systems + controls = 25,250 requests, about 2 x 45 min with `SHARD_N=2`.
  Do not use 48 GB cards at TP2 (degenerate "!!!!" output with this vLLM/MXFP4 build; the sanity gate stops the job).
Outputs: `judge/units/*.jsonl`, `judge/ext_judge_<N>.json` (+ `.perpage.jsonl`), `judge/sanity_50.json`.

## Driver
`bash run_all.sh <page_table> <student_outputs> [out_dir]` checks coverage, then submits A (CPU), J (judge array, after
A) and G (aggregation + `make_extraction_quality.py`, after J). Default out dir `$WORK_DIR/eval/extraction/runs/n<N>`.
Final file: `<out_dir>/extraction_quality.json` = `{pages, source, ext_vs_dripper, judge}` = the figure data file `extraction_quality.json` of `analysis/data`.

Numbers of our release-decoding run (5,000 pages): token F1 vs Dripper (macro) student 92.3, resiliparse 76.5,
trafilatura 73.2, jusText 59.3; judge total /6 student 5.62, Dripper 5.60, resiliparse 4.99, trafilatura 4.86,
jusText 4.70 (25,000/25,000 judgments parsed).

## Environment
`RESCRAPER_ROOT`, `WORK_DIR`, `HF_HOME`; `REFINER_PY` (trafilatura 2.0.0, jusText 3.0.2, numpy), `DCLM_PY` (resiliparse
0.16.0), `JUDGE_PY` (vLLM able to serve gpt-oss-120b); `HTML`, `SHARD_N`, `EXT_PROMPT`, `JUDGE_MODEL`, `JUDGE_DATE`,
`MAXCH`, `GPU_UTIL`. SLURM partitions are placeholders (`GPU_PARTITION` / `CPU_PARTITION`).
