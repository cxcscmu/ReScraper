# evaluation/heldout_set/

Builds `heldout5k.jsonl`: the 5,000-page held-out table behind every Section-5 analysis (and Figure 1), labelled by the
teacher cascade with the released student's rewrite rule (`edu1`), plus `html5k.jsonl` (the raw HTML of every page).
Its first 960 rows are `heldout960`, the earlier validation prefix.

## Teacher cascade and rule (`edu1`)
Dripper extraction (the pool Dripper run, `$DRIPPER_STEP1_DIR` / `$DRIPPER_STEP2_DIR`; stage-1 `rm` ops via
`lib/editops.make_ops` on the `<lid:n>` rendering) -> Qwen3.8-27B strict-subset keep/edit/delete (the SFT teacher:
`prompts/teacher_refine_qwen27b.txt`, greedy, thinking off, `max_tokens = min(32768 - prompt, text + 256)`) ->
teacher-deleted pages with FineWeb-Edu >= 1.0 become `<rewrite>` + a T=1 RePro-1B paraphrase (`lib/repro_olmo.py`:
T 1.0, top-p 0.9, 2,048 tokens, 7,000-character chunks; `lib/rwclean.py` clean + `is_dirty` + word ratio 0.2-3.0; no
acceptance filter); otherwise `<delete>`. FineWeb-Edu = the `text_decisions` value of the joined pool page (what the
Stage-2 SFT build reads), else the same classifier recomputed on the same Dripper text (CPU fp32, batch 64, 512 tokens).
Paraphrase = the pool's `two_stage_pool/rwrepro_olmo_temp1` text of the joined page, else a newly generated one.
The RePro 1B rephraser is the public RePro 1B rephraser checkpoint (Yu et al., 2025); identifier withheld for anonymity
(`RW_MODEL`).

## Rows
| rows | `source` | how |
|---|---|---|
| 0..959 | `heldout960` | first 960 rows of the earlier 3,161-page staged held-out table, relabelled with `edu1` (`heldout960/`) |
| 960..3160 | `relabel2201` | the rest of that table: keep/edit copied byte for byte, every teacher-deleted page re-decided as above |
| 3161..4999 | `new_shard` | two new held-out pool shards through the same cascade, in (shard draw order, step-2 row) order until 5,000 |

The 3,161-page staged table (`$HELDOUT3161_DIR`: `render_*.jsonl` + `sft_e2eC_gold_tagged.jsonl`) covers the three
held-out pool shards of the Dripper SFT runs, labelled by the same Dripper + Qwen3.8-27B cascade; it is an input here
(its builder predates this release; `s1_old_rows.py` re-derives every row's raw record and re-renders its input, 3,161/3,161
byte-identical).

## Held-out guarantee (by shard stem)
Excluded when drawing the two new shards (`s2a_new_shards.py`): the 1,109 Stage-1 SFT shards of the released student
(`$STAGE1_SFT_STEMS`), 2 known-bad pool shards, every pool shard that holds a page of the two-stage refiner's
50,000-page SFT source (mapped by WARC-Record-ID over all 10,319 resiliparse shards, `s1b_two_stage_shards.py`:
3,962 shards), the 103 shards of the other pool SFT/held-out files, and the 3 existing held-out shards. Of the 10,316
pool shards with every artifact, 5,617 remain; `random.Random(5000).sample(sorted(candidates), 2)` picks
`CC-MAIN-20180220224819-20180221004819-00207` and `CC-MAIN-20190422195208-20190422221208-00212`. A raw-WARC check finds
0 pages of the two-stage source in any of the 5 held-out shards:
`CC-MAIN-20140707234035-00041-ip-10-180-212-248.ec2.internal` (rows 0..959 and part of 960..3160),
`CC-MAIN-20170924171658-20170924191658-00275`, `CC-MAIN-20200709002952-20200709032952-00415` (rest of 960..3160),
`CC-MAIN-20180220224819-20180221004819-00207`, `CC-MAIN-20190422195208-20190422221208-00212` (3161..4999). Identical page texts within the new candidates, or equal
to an existing row, are dropped (29 + 2 pages), so no text enters the table twice.

## Steps (`bash run_all.sh` submits all of them)
| step | script | compute (our run) | output |
|---|---|---|---|
| 960 | `heldout960/heldout960.sbatch`: `join_pool960.py` -> `edu_score960.py` -> `need960.py` -> `gen_repro_t1_960.py` -> `build_heldout960.py` | 1 GPU (48 GB), ~1 min generation | `$HELDOUT960_DIR/sft_e2eC_heldout960_edu1_tagged.jsonl` (keep 295 / edit 286 / delete 351 / rewrite 28) |
| 1 | `s1_old_rows.sbatch` -> `s1_old_rows.py` | 8 CPUs, 12 GB, ~20 min | `work/old_prov.jsonl` (provenance, pool join, edu), `html_old.jsonl`, `two_stage_src_warc.tsv`, `other_sft_shards.json` |
| 1b | `s1b_two_stage_shards.sbatch` -> `s1b_two_stage_shards.py` | 16 CPUs, ~10 min | `work/two_stage_refiner_stems.txt` |
| 2 | `s2_new_shards.sbatch`: `s2a_new_shards.py` (exclusion, pick, render, SFT seed filter) -> `s2b_lid.py` (fastText, `PIPE_PY`) -> `s2c_new_candidates.py` (filters, dedup, pool join, edu) | 8 CPUs, 12 GB, ~15 min | `work/pick.json`, `cand_new.jsonl` (27B input), `cand_new_meta.jsonl`, `cand_new_full.jsonl`, `html_new.jsonl` |
| 3 | `s3_teachers.sbatch`: `data_construction/teacher_refine/label_qwen27b.py` (8 ranks, `T27_SRC=work/cand_new.jsonl`, `T27_DST=work/t27`) -> `s3_need.py` -> `s3_gen_repro.py` | 1 node, 8x H200 (141 GB); ~10 min | `work/t27/part-*-of-008.jsonl`, `repro_need.jsonl` (69 pages), `repro_out.jsonl` |
| 4 | `s4_assemble.sbatch`: `s4_build.py` -> `s5_resiliparse_join.py` -> `s6_html_table.py` -> `s7_stats.py` | 2 CPUs, a few minutes | `$HELDOUT5K_DIR/heldout5k.jsonl`, `html5k.jsonl`, `build/sft_e2eC_heldout5k_edu1_tagged.jsonl` (+ `_val_idx.json`), `build/stats.txt` |

`s4_build.py` asserts that rows 0..959 are byte-identical to the `heldout960` table. The 27B labeling script is the
Stage-1 SFT teacher script (`data_construction/teacher_refine/`), run with its source/destination set through
`T27_SRC` / `T27_DST` and 8 ranks (the SFT run used 128); decoding is greedy, so a page's label does not depend on the
striping. Pages the 27B script skips (empty, or prompt > 32,704 tokens) have no label and are not in the table.
T=1 paraphrases are single samples and not bit-reproducible.

## Row fields of `heldout5k.jsonl`
`gid` (0..4999), `source`, `stem`, `step1_idx` (line of the Dripper step-1 shard), `in_md5` (md5 of the raw HTML =
`html5k.jsonl`), `url`, `warc_id`, `input` (the numbered `<lid:n>` rendering the student reads), `drip` (Dripper text),
`output` (teacher program, staged format), `tag` (keep/edit/delete/rewrite), `gfinal` (teacher final text =
`e2e_ops.body_from_prediction_dfirst(input, output)`, empty for delete), `edu`, `edu_source`, `edu_recomputed`,
`old_tag` (label before the relabel, rows < 3161), `paraphrase_source`, `pool_idx`, `pool_student_kept` (the two-stage
refiner kept the page in the pool run), `resiliparse` (the pool resiliparse_extract text of the page, the input of every
baseline), `resiliparse_join`, `resiliparse_row`; new-shard rows also carry `step2_row, cid, t27_tag, staged_kind,
t27_finish`; older rows carry `simp_row, render_i, rerender_eq_input`.

Label mix (our build): all 5,000: keep 1,881 (37.6%), edit 1,336 (26.7%), delete 1,578 (31.6%), rewrite 205 (4.1%).
Resiliparse join: 5,000/5,000 by WARC-Record-ID, no empty text.

## Environment
`WORK_DIR`, `RESCRAPER_ROOT`, `HF_HOME`, `DCLM_DIR` (fastText LID model; or `LID_MODEL`), `RW_MODEL`, `TEACHER_MODEL`
(read by the 27B script), interpreters `REFINER_PY` (vLLM + dripper renderer + transformers), `PIPE_PY` (fasttext),
`TEACHER_PY`. Inputs (defaults in `evaluation/eval_paths.py`): `HTML_POOL_DIR`, `POOL_SIMP_DIR`, `DRIPPER_STEP1_DIR`,
`DRIPPER_STEP2_DIR`, `RESILIPARSE_EXTRACT_DIR`, `TWO_STAGE_POOL_DIR`, `TEXT_DECISIONS_DIR`, `SFT_DIR` / `SFT_DATA_DIR`,
`HELDOUT3161_DIR`, `STAGE1_SFT_STEMS`, `TWO_STAGE_SFT_SOURCE_GLOB`. Outputs: `HELDOUT960_DIR`, `HELDOUT5K_DIR`
(default `$WORK_DIR/eval/heldout960`, `$WORK_DIR/eval/heldout5k`). SLURM partitions are placeholders
(`GPU_PARTITION` / `CPU_PARTITION`, or `-p`).
