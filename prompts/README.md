# prompts/

Every prompt used in the paper, verbatim (byte copies of the files the runs read).

| file | used by | notes |
|---|---|---|
| `teacher_refine_qwen27b.txt` | refining teacher, Qwen3.8-27B (`data_construction/teacher_refine/label_qwen27b.py`); also the held-out labels | system prompt; user message = Dripper's extracted text (no line ids). Everything up to the task line is the FineWeb-optimized refinement prompt published with UltraX; the seven examples are those of UltraX's released base prompt (they contain full-width punctuation, kept verbatim). Deletion marker: `[Content valueless, deleted]`. Greedy, thinking disabled |
| `repro_rewrite_system.txt`, `repro_rewrite_user.txt` | RePro 1B rephraser (`lib/repro_olmo.py`, `data_construction/rescue/`) | upstream RePro recycling prompt; `{TEXT}` is replaced by a 7,000-character chunk of the Dripper text; output = text after "Here is a paraphrased version:". T=1.0, top-p 0.9, 2,048 tokens |
| `student_system.tmpl` | template of the student system prompt | `__FREQS__` is filled by `data_construction/sft_sets/finalize_prompt.py` with the operation frequencies of the SFT file being trained |
| `student_system_stage1.txt` | Stage-1 training | frequency line `<keep> 38%, <edit> 26%, <delete> 36%, <rewrite> 0.1%` |
| `student_system_stage2.txt` | Stage-2 training and all pool / held-out inference of the released model | frequency line `<keep> 27%, <edit> 19%, <delete> 25%, <rewrite> 30%`; the two files differ only in that line. Its RESCUE paragraph describes the three-band Stage-1 policy (see `data_construction/README.md`) |
| `two_stage_refiner_system.txt` | the two-stage refiner ablation (`ablations/two_stage_refiner/`) | the refinement rules of the teacher prompt, framed for text input |
| `judge_extraction.txt` | gpt-oss-120b extraction judge (Figure 8, prompt printed in Appendix "Extraction judge") | `evaluation/extraction/` |
| `judge_keep_or_drop.txt` | gpt-oss-120b keep-or-drop judge (Figure 1) | `evaluation/judges/` |

Dripper's own inference prompt is part of the Dripper (MinerU-HTML) release and is not duplicated here. DataMan and
FineWeb-Edu are classifiers used with their released heads.
