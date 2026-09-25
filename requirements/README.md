# Environments

| file | used by | notes |
|---|---|---|
| `refiner.txt` | `data_construction/` (rendering, seed sampling, SFT set building, RePro rewrites, FineWeb-Edu), `training/`, `inference/`, `ablations/`, most of `evaluation/` (incl. the gpt-oss-120b judges) | Python 3.12, CUDA 12.8, torch 2.9.0, transformers 4.57.6, vLLM 0.11.1, DeepSpeed, flash-attn 2.8.3; `dripper==1.0.0` (MinerU-HTML) provides the page renderer |
| `teacher.txt` | `data_construction/teacher_refine/` | an environment able to serve Qwen3.8-27B (vLLM 0.28.0, transformers 5.16.1) |
| `dclm.txt` | `corpus/tokenize.sh`, `pretraining/` | DCLM training/evaluation environment (Python 3.10, torch 2.3.1, open_lm 0.0.34); DCLM data processing (BFF dedup, `tokenize_shuffle.py`, rule mappers) uses the dependencies of the DCLM repository itself |
| `tools.txt` | `baselines/`, `evaluation/` | datatrove 0.2.0 (FineWeb-rule, separate environment `DATATROVE_PY`), resiliparse 0.16.0, trafilatura 2.0.0, jusText 3.0.2 |

The gpt-oss-120b judges run in a vLLM 0.11.1 environment (`JUDGE_PY`, the refiner environment works). Further
details are in the README of each directory.
Set the interpreters in `configs/paths.env.example` (`REFINER_PY`, `TEACHER_PY`, `DCLM_PY`, `PIPE_PY`, ...).
