# evaluation/judges/

The keep-or-drop judge and the rule flags behind Figure 1 (the motivation figure): (a) the share of the pages each type
of rule drops that an independent LLM judges worth keeping, (b) keep-drop accuracy of each pipeline against its 1B Core.

## Keep-or-drop judge (`keep_judge.py`, `keep_judge.sbatch`)
- `openai/gpt-oss-120b`, system prompt `prompts/judge_keep_or_drop.txt` (verbatim in the appendix), user message
  `"PAGE:\n<<<\n" + page + "\n>>>"`, page = the rendered page the student reads without line ids (field `rendered` of
  `ext_texts_5000.jsonl`, written by `evaluation/extraction/ext_metrics.py`), truncated to 24,000 characters (+
  `"\n[... truncated for length]"`) and shrunk in 20% steps if the prompt would not fit. The judge never sees any
  pipeline's output.
- vLLM `LLM.chat`, temperature 0, max_tokens 2048, `reasoning_effort=low`, max_model_len 32768, chat-template date pinned
  (`JUDGE_DATE`); parsing = last JSON object with `value` and `verdict` (after the `assistantfinal` marker if present);
  the job fails if < 99% parse.
- 1x 96 GB GPU (RTX PRO 6000 Blackwell), 7 min for 5,000 pages (5,000/5,000 parsed, none hit the length limit).
- Output `keep_judge_5000.jsonl` `{gid, parsed: {value, verdict, reason}, raw, finish_reason, ...}`. Figure 1 uses only
  `verdict` (3,495 keep / 1,505 remove).

`sbatch keep_judge.sbatch $WORK_DIR/eval/extraction/runs/heldout5k_rel/ext_texts_5000.jsonl $WORK_DIR/eval/keep_judge/keep_judge_5000.jsonl`
(`analysis/analyze_rule_groups.py` reads `rule_flags_5000.jsonl` from here.)

## Figure 1(b): `rule_motivation.py`
`python rule_motivation.py heldout5k.jsonl <heldout_pipeline>/work/heldout5k_rel keep_judge_5000.jsonl rule_motivation.json`
(CPU, seconds). Per system (RefinedWeb-rule, FineWeb-rule, ProX-C, UltraX, the teacher cascade, ReScraper with release
decoding; a page is kept when the system emits non-empty text): TP/FP/FN/TN against the judge, recall = worth-keeping
pages kept, and TN/(TN+FP) = junk pages dropped; keep-drop accuracy = their mean (our run, % kept / % dropped:
ReScraper 87.90 / 68.97, UltraX 96.85 / 40.07, RefinedWeb-rule 67.93 / 60.86, ProX-C 94.39 / 32.29, FineWeb-rule
50.41 / 73.89). Also per drop reason of each rule stack. `analysis/plot_keepdrop_vs_core.py` plots them
against the 1B Core scores.

## Figure 1(a): rule flags (`allrules/`)
`sbatch allrules/run.sbatch heldout5k.jsonl <heldout_pipeline>/work/heldout5k_rel keep_judge_5000.jsonl $WORK_DIR/eval/keep_judge`
(CPU, 4 cores, < 1 h) runs
- `rw_allrules.py` (`DCLM_PY`): RefinedWeb-rule with every rule evaluated independently (a filter that drops the page is
  recorded and the next rule is tested on the unfiltered page; line modifiers run as in the stack, then
  `word_removal_ratio_filter`) -> `rw_flags.jsonl`;
- `fw_allrules.py` (`DATATROVE_PY`): FineWeb-rule (datatrove 0.2.0) with the checks inside each filter run without early
  return (the filter source is rewritten mechanically so every `return False, <reason>` is recorded); c4 still rewrites
  the text and the FineWeb-quality checks run on the rewritten text -> `fw_flags.jsonl`;
- `build_rule_flags.py` -> `rule_flags_5000.jsonl`, read by `analysis/analyze_rule_groups.py` (merged rule groups of
  `analysis/plot_rule_worth_keeping.py`).
Same code, arguments and inputs as the stack runs of the held-out pipeline (`heldout_pipeline/lib/rw_rule_chain.py`,
`fw_rule_chain.py`). Check: on all 5,000 pages the first failing rule in stack order equals the stack's recorded drop
reason, and "no rule fails" equals "kept" (0 mismatches for both stacks).

### Columns of `rule_flags_5000.jsonl` (one row per page, joined on `gid`)
- `judge_verdict`: `keep` or `remove`, from `keep_judge_5000.jsonl` (3,495 keep).
- `rescraper_op`: `keep`, `edit`, `rewrite`, `delete`, or `skip:too_long` (11 pages over the context), release decoding.
- `rescraper_deletes`: ReScraper emits no page (op `delete` or `skip:too_long`; 1,461 pages).
- `refinedweb_rule_kept` / `fineweb_rule_kept`, `refinedweb_rule_first_reason` / `fineweb_rule_first_reason`: actual
  outcome of each stack (the stacks stop at the first failing rule); the first reasons define the groups of Figure 1(a).
- Per-rule flags (true when that rule fails on the page, evaluated independently), with their Figure 1 labels:
  `rw_massive_web_repetition_filters` repeated lines / n-grams; `rw_word_removal_ratio_filter` cleaning cut > 5% -> page
  dropped; `rw_page_length_filter` page too short; `rw_alphabetic_word_ratio_filter` few alphabetic words;
  `fw_gopher_qual_gopher_below_alpha_threshold` < 80% of words have a letter; `fw_fineweb_char_dup_ratio` duplicated
  characters; `fw_gopher_rep_dup_line_frac` duplicate lines; `fw_fineweb_line_punct_ratio` few lines end in punctuation;
  `fw_gopher_rep_duplicated_5_n_grams` duplicated 5-grams; `fw_lang_None` not detected as English;
  `fw_gopher_qual_gopher_short_doc` document too short.
- `refinedweb_rule_all_fired` / `fineweb_rule_all_fired`: every rule that fails, in stack order.
- `fineweb_rule_eval_incomplete`: set on 6 pages where a later FineWeb-quality check divided by zero because c4 had
  emptied the text, so only the checks before it were evaluated.

## Environment
`RESCRAPER_ROOT`, `WORK_DIR`, `HF_HOME`, `DCLM_DIR`, `NLTK_DATA`; `JUDGE_PY`, `DCLM_PY`, `DATATROVE_PY`, `REFINER_PY`; `JUDGE_MODEL`,
`JUDGE_DATE`, `MAXCH`, `MAX_TOKENS`, `GPU_UTIL`. SLURM partitions are placeholders (`GPU_PARTITION` / `CPU_PARTITION`).
