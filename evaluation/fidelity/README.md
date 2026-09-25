# evaluation/fidelity/

Teacher fidelity of ReScraper on the held-out pages: how closely the 0.6B student follows the teacher cascade
(Section 5, "Effectiveness of learning the teachers' behaviors"; Figure 7 and Table `tab:fidelity-by-length`).

| file | what |
|---|---|
| `teacher_metrics.py` | library used by the held-out pipeline (`heldout_pipeline/stages/s1_ours_exec.py`, `s1_rel_exec.py`): `teacher_metrics()` (decision accuracy over 4 operations and keep-or-delete accuracy over the pages sent to the model; pooled token P/R/F1 = overlap of lower-cased `\w+` token multisets of the text each page contributes, deleted pages -> `""`, on pages whose prompt + teacher target fit the 32,768-token context; 95% CIs from 2,000 page-bootstrap resamples; 4x4 confusion), `rows_for()` / `rows_from_executed()` (teacher vs student text per page), `decision_flow()` (teacher operation x student operation over all pages; skipped / unparseable pages count as delete) |
| `score_fidelity.py` | stand-alone CLI on a page table + an executed `ours.jsonl`: `python score_fidelity.py heldout5k.jsonl work/heldout5k_rel/ours.jsonl fidelity.json` |

Teacher text per page = `lib/e2e_ops.body_from_prediction_dfirst(input, output)` of the teacher program (`gfinal`);
student text = the executor output of its program (`text`); both are `""` for `<delete>`.

On the release-decoding run (4,989 of 5,000 pages sent, 3 of them too long for a token score) this gives decision
accuracy 74.80, keep-or-delete 89.32, token P / R / F1 89.11 / 89.41 / 89.26, and the decision flow of
the paper's `decision_flow.json` exactly (also written by `s1_rel_exec.py` as `decision_flow_5k_rel.json`).
The per-length-quartile table of the appendix (`fidelity_by_length_5k.json`) is computed by
`analysis/analyze_fidelity_by_length_5k.py` from the same two files with the same definitions.

CPU only; about a minute for 5,000 pages. Needs the repository's `lib/` (imported by relative path).
