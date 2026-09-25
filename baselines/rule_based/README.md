# baselines/rule_based/

Rule-based cleaning baselines, all run on resiliparse-extracted text:
- RefinedWeb-rule: DCLM's RefinedWeb heuristics.
- C4-rule: DCLM's `c4.yaml`.
- FineWeb-rule: datatrove's FineWeb filters.

The resiliparse settings are `extract_plain_text(HTMLTree.parse(html), main_content=True, alt_texts=False,
preserve_formatting=True)`. DCLM's `resiliparse_extraction_modifier` uses them, and so does every page-level
runner here. Pages whose extraction is empty are dropped.

## Files
| file | what it does | feeds |
|---|---|---|
| `run_resiliparse_pipeline.sh` | See the pipeline section below. | main table RefinedWeb-rule (7.04B unique tokens); Figure 3 resiliparse bar; the input of the UltraX baselines |
| `page_rules.py` | Runs the RefinedWeb-rule and C4-rule stacks page by page with DCLM's own mapper factory (`get_mapper(..., _safe=True)`, an exception drops the page), behind DCLM's lang+length gate. See the C4 note below. | page-level baseline outputs |
| `fineweb_rule/fw_rule.py`, `fw_rule.sbatch` | FineWeb-rule: see the section below. | FineWeb-rule sample; the held-out pipeline runs it page by page (`evaluation/heldout_pipeline`) |

`page_rules.py` notes:
- C4-rule is DCLM `c4.yaml` with its two corpus-level `exact_dedup` steps removed. Every per-page step is kept,
  including `langdetect` en >= 0.99 and `within_page_dedup`.
- The nltk `punkt_tab` tokenizer is patched in for nltk >= 3.9.
- Input rows: `{k, resiliparse, metadata, url, in_langlen_pool}`.
- Output rows: `{k, gate, rw, c4, rw_reason, c4_reason, ...}`.

### `run_resiliparse_pipeline.sh`
It orchestrates the pipeline from a submit host and polls `squeue` between phases:
1. Resiliparse extraction of the pool. Config `resiliparse_extract.yaml`; the output is
   `$OUT_ROOT/resiliparse_extract` (18,025,558 -> 18,001,982 pages).
2. RefinedWeb rules, config `dclm_baseline_refinedweb_post_lang.yaml`, output `$OUT_ROOT/step3b_post_lang`
   (-> 8,894,137). The stack is:
   - `page_length_filter` (50 words or more);
   - word-length, symbol, bullet, ellipsis and stop-word filters;
   - Gopher repetition filters;
   - the line modifiers;
   - `word_removal_ratio_filter`.
3. BFF dedup (`corpus/dedup_bff.sh refinedweb_rule`, -> 7,497,735 documents).
4. Tokenization (`corpus/tokenize.sh refinedweb_rule`).

The language filter (DCLM step3a) is not re-run: the pool was already language-filtered upstream. Phases 1 and 2
each fan out to 40 CPU jobs (32 CPUs, 200 GB, `process_no_ray.py`).

### FineWeb-rule (`fineweb_rule/`)
- Input: the lang+length-filtered HTML pool (`$FW_SRC`), 60 randomly chosen shards.
- Steps:
  1. datatrove `URLFilter`.
  2. resiliparse extraction.
  3. Blank lines removed. Trafilatura's layout has none, and the Gopher line rules would otherwise count them.
  4. `LanguageFilter` (fastText lid.176, en, 0.65).
  5. `GopherRepetitionFilter`.
  6. `GopherQualityFilter`.
  7. `C4QualityFilter(filter_no_terminal_punct=False)`.
  8. `FineWebQualityFilter`.
- Filter order and arguments are those of datatrove's `examples/fineweb.py`. MinHash dedup and PII formatting are
  not applied.
- Output: surviving shards, a 5,000-document sample (`fineweb_rule_5k.jsonl`) and per-filter counts
  (`fw_counts.json`).

## Environment
- `run_resiliparse_pipeline.sh`: `WORK_DIR`, `DCLM_DIR`, `RESCRAPER_ROOT`, `HTML_POOL_DIR`, `PIPE_PY`,
  `CPU_PARTITION`, and optionally `RESILIPARSE_OUT` (default `$WORK_DIR/dclm_pipeline/resiliparse`).
  - `DCLM_DIR` is upstream `mlfoundations/dclm` with `pretraining/dclm_patches` applied, which adds the modifier,
    the configs and `process_no_ray.py`.
  - `PIPE_PY` needs DCLM's data-processing requirements plus `resiliparse`.
- `page_rules.py`: run with cwd `$DCLM_DIR` (or set `DCLM_DIR`) under `PIPE_PY`. Needs DCLM's `c4.yaml` (upstream),
  `dclm_baseline_refinedweb_lang_only.yaml` and the `_post_lang*.yaml` configs.
- `fw_rule.py`: `DATATROVE_PY` = datatrove 0.2.0 + resiliparse + fasttext. It reads `LID_MODEL` (default: the
  lid.176 copy in DCLM), `HTML_POOL_DIR` / `FW_SRC`, and `FW_OUT` in the sbatch. datatrove's `URLFilter` downloads
  its block lists (`HF_HOME`), and the Gopher filters need nltk punkt data (`NLTK_DATA`).

## Compute
All CPU.
- RefinedWeb rules over the pool: about 6 h wall on 40 x 32-CPU jobs.
- Dedup and tokenization: see `corpus/`.
- `fw_rule.sbatch`: one 16-CPU job, under 2 h.

## Not included
The runs that built the full-pool C4-rule (5.06B) and FineWeb-rule (5.16B) corpora of the main table are not part
of this release. `page_rules.py` and `fw_rule.py` implement the same rule stacks, at page level and on a pool sample
respectively.
