# lib/

Shared modules imported by the scripts in this repository (each script prepends this directory to `sys.path`).

| module | what it provides |
|---|---|
| `editops.py` | `number_lines` (the `<lid:n>` line numbering of the model input), `make_ops` (diff a strict-subset edit into `rm A`, `rm A-B`, `sub N: "s"` operations), `apply_ops` (execute operations on a numbered page) |
| `e2e_ops.py` | the serialized target and the deterministic executor: `to_staged_row` (compose Dripper removals + the refining teacher's decision into one target), `parse_staged`, `body_from_prediction_staged`, `body_from_prediction_dfirst` (decision-first reader used for the released model). `python lib/e2e_ops.py` runs a self-test. |
| `pool_join.py` | locations of the Dripper step-1/step-2 outputs (`DRIPPER_STEP1_DIR`, `DRIPPER_STEP2_DIR`) and the ordered head-of-text check that aligns step-2 rows with step-1 rows |
| `repro_olmo.py` | RePro rephraser prompt, conversation builder, 7,000-character chunking, output extraction, sampling parameters (T=1.0, top-p 0.9, 2,048 tokens) |
| `rwclean.py`, `rwstrict.py` | filters for rephraser artefacts (meta-commentary, prompt echo, reasoning traces) in generated rewrites |
| `dom_annot.py` | maps rendered lines to the HTML block element they came from (used by the Stage-1 rescue rule to find prose blocks) |
| `prox_chunk_utils.py` | vendored ProX executor for chunk-level refining programs (ProX-C baseline) |

Decision tags and executed provenance tags (`e2e_tag`) are different names for the same event:
`<keep>` -> `<extract>`, `<edit>` -> `<refine>`, `<delete>` -> `<delete>` (no document), `<rewrite>` -> `<rewrite>`.
