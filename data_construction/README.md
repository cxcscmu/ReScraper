# data_construction/

Builds the supervised targets of ReScraper from three teachers run in sequence on the same line-numbered rendering
of each page (paper Section 3.2): Dripper (`<extract>`), Qwen3.8-27B (`<keep>` / `<edit>` / `<delete>`) and the
RePro 1B rephraser (`<rewrite>`), with the FineWeb-Edu classifier deciding which deleted pages are rescued.

```
DCLM pool shard (raw HTML)
  │ dripper/                         Dripper step 1 (main-content HTML) + step 2 (text)   -> step1_inference/, step2_text/
  ├─ seed_pages/sample_seed_pages.py SFT seed pages: Dripper text of ~1.6M pages (held-out shards excluded)
  ├─ teacher_refine/label_qwen27b.*  Qwen3.8-27B refinement of every seed page (strict-subset prompt, greedy)
  ├─ base_set/                       tag rule -> render the raw HTML (lib/render.py) -> <lid:n> numbering ->
  │                                  serialized target (Dripper rm-ops + teacher decision on the same ids) ->
  │                                  decision-first -> every <rewrite> folded back to <delete>
  │                                  => base set, 1,383,115 rows (513,724 keep / 359,292 edit / 510,099 delete)
  ├─ rescue/                         FineWeb-Edu on the deleted pages of the pool (+ Stage-1 rescue actions),
  │                                  RePro 1B rewrites (T=1.0, top-p 0.9) of deleted pages with int_edu >= 1
  └─ sft_sets/                       Stage-1 set (base set + Stage-1 rescue relabel) and Stage-2 set
                                     (every teacher-deleted page with FineWeb-Edu >= 1.0 and a rewrite -> <rewrite>,
                                     rewrite raised to 30% by subsampling the other operations)
```

Target format (decision-first, `lib/rescraper_ops.py`):
```
<keep>|<edit>|<delete>|<rewrite>     line 1: the operation
<extract>                            line 2, always literal
rm A / rm A-B ...                    Dripper's whole-line removals (may be empty)
<same operation again>               stage boundary
<payload>                            <edit>: rm N / rm A-B / sub N: "s" (deletion only); <rewrite>: the new text
```

## Steps

All scripts read paths from `configs/paths.env` (`WORK_DIR`, `SFT_DIR`, `DRIPPER_STEP1_DIR`, `DRIPPER_STEP2_DIR`,
`HELDOUT_SHARDS`, `RW_MODEL`, `TEACHER_MODEL`, python interpreters). SLURM partitions are placeholders.

| # | script | what it does | compute (our run) |
|---|---|---|---|
| 0 | `dripper/` | Dripper over the pool (see `dripper/README.md`) | 149.6 H200 GPU-h attributable to the SFT pages |
| 1 | `seed_pages/sample_seed_pages.py <target_n>` | sample SFT seed pages from `step2_text` (English >= 0.65, no residual markup, text-hash dedup; held-out shards excluded by stem) -> `$SFT_DIR/seed/seed_pages.jsonl` | CPU, 48 processes |
| 2 | `teacher_refine/label_qwen27b.sbatch` | Qwen3.8-27B with `prompts/teacher_refine_qwen27b.txt` on every seed page (greedy, thinking off, `max_tokens = min(32768 - prompt, text + 256)`) -> `$SFT_DIR/teacher_refine/part-*.jsonl` | 16 nodes x 8 GPUs (WORLD 128); 203.9 GPU-h attributable to the SFT pages |
| 3a | `base_set/select_pages.py <dir>` | apply the tag rule (deleted -> `<delete>`, refined == Dripper text up to whitespace -> `<extract>`, else `<refine>`), group by shard -> `<dir>/wanted.jsonl` | CPU |
| 3b | `base_set/join_render.sbatch <dir>` | raw HTML via Dripper step 1 -> canonical rendering (`webkit_txt`, 60 s timeout) -> `<dir>/join_*.jsonl` | 32 CPU tasks |
| 3c | `base_set/build_base_set.sbatch <out> <dir>...` | `build_staged.py` -> `build_dfirst.py` (+ `verify_dfirst.py`) -> `build_norw.py` -> `check_stem_disjoint.py` -> `merge_gate.py union` | CPU, < 4 h |
| 4 | `ablations/two_stage_refiner/` | the two-stage refiner over the pool: `two_stage_pool/input` (a link to `$DRIPPER_STEP2_DIR`) -> `two_stage_pool/best`; used as the pool-scale proxy for the teacher's deletions in step 5 | 8-GPU nodes |
| 5a | `rescue/rescue_build.sbatch` | FineWeb-Edu of every page the two-stage refiner deleted, plus the Stage-1 rescue actions -> `rescue_add/text_decisions/<shard>.jsonl` | 14 nodes x 8 GPUs |
| 5b | `rescue/score_edu_pool.sbatch` | FineWeb-Edu `int_edu` sidecar -> `two_stage_pool/edu/` | 32 x 1 GPU |
| 5c | `rescue/rewrite_pool.sbatch` | RePro 1B rewrites of deleted pages with `int_edu >= 1` -> `two_stage_pool/rwrepro_olmo_temp1/` (2,848,492 rewrites available to the SFT build) | 4 nodes x 7 GPUs; 6.9 GPU-h attributable to the SFT pages |
| 6 | `sft_sets/build_stage1_set.py`, `build_stage2_set.py`, `finalize_prompt.py` | run by `training/stage1_chain.sbatch` and `training/stage2_chain.sbatch` | CPU |

### Resulting SFT sets (paper Table `tab:sft-composition`)

| | Stage 1 | Stage 2 |
|---|---:|---:|
| `<keep>` | 522,312 | 35,189 |
| `<edit>` | 359,292 | 24,611 |
| `<delete>` | 499,924 | 32,239 |
| `<rewrite>` | 1,587 | 39,445 |
| total | 1,383,115 | 131,484 |

Stage 1 = the base set relabelled with the Stage-1 rescue actions of `rescue_build.py` (FineWeb-Edu >= 1.5 -> `<keep>`
verbatim; 1.0 <= edu < 1.5 with a prose block of >= 150 words and an accepted paraphrase -> `<rewrite>`).
Stage 2 = scheme `edu1` of `build_stage2_set.py`: every base-set `<delete>` row whose FineWeb-Edu score (sidecar
`edu`, computed on the Dripper text) is >= 1.0 and for which a rewrite exists becomes `<rewrite>` (relabel counts:
39,445 flipped, 398,008 unchanged, 72,646 without a score); the rewrite class is then raised to 30% by subsampling the
other classes stratified by tag (seed 17). No DataMan gate and no acceptance test on the paraphrase; rewrites are
dropped only when empty, flagged by the artefact filter (`lib/rwclean.py`), or outside a 0.2-3.0 word-length ratio.

## Notes and deviations (read before reproducing exactly)

- **Two halves.** The base set was assembled from two halves of the seed pages (~1.0M, then the ~0.62M complement,
  `select_pages.py` with `PREV_WANTED`). The first half's targets were derived through an intermediate file that also
  carried older rewrite targets (built with an earlier rewriting recipe that is not part of this release);
  `build_norw.py` folds every such rewrite back into `<delete>`, and `select_pages.py` verifies on a 4,000-page
  sample that it reproduces the first half's non-rewrite targets byte for byte (>= 95% required). The first half
  additionally went through `leak_filter.py` (a content-collision filter against held-out pages); held-out
  disjointness itself is by shard stem.
- **Held-out shards** are excluded by stem at seed sampling and checked again at the end
  (`check_stem_disjoint.py`); identical text in different shards is a natural duplicate and is not filtered.
- **The two-stage refiner decides which pool pages are candidates for rescue** (step 4): the FineWeb-Edu sidecar and
  the rewrites cover the pages it deleted. At label time the Stage-2 builder only uses pages the 27B teacher deleted.
- **Stage-2 system prompt.** Both stages are trained with `prompts/student_system.tmpl` filled by
  `finalize_prompt.py`; the template's RESCUE paragraph describes the three-band Stage-1 policy, which the Stage-2
  targets (single threshold at 1.0, no acceptance test) do not follow. The prompts are released verbatim.
- Stage 1 contains a small `<rewrite>` share (0.1%); the paper's Stage-1 column reports the trained file.
