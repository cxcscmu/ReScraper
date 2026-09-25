# Sourced by every evaluation SLURM job / driver: default locations, interpreters and runtime flags.
# Every value can be overridden from the environment (see configs/paths.env.example). Same names and defaults as
# evaluation/eval_paths.py.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${RESCRAPER_ROOT:?set RESCRAPER_ROOT to the repository root (see configs/paths.env.example)}"
export EVAL_DIR=${EVAL_DIR:-$WORK_DIR/eval}
export HELDOUT5K_DIR=${HELDOUT5K_DIR:-$EVAL_DIR/heldout5k}
export HELDOUT960_DIR=${HELDOUT960_DIR:-$EVAL_DIR/heldout960}
export HELDOUT_PIPELINE_DIR=${HELDOUT_PIPELINE_DIR:-$EVAL_DIR/heldout_pipeline}
EV=$RESCRAPER_ROOT/evaluation

# interpreters (conda/venv python binaries); plain `python` if unset
PY=${REFINER_PY:-python}          # vLLM 0.11.1 / transformers 4.57 env (student, ProX-C, RePro, scorers, gpt-oss-120b)
PYTEACHER=${TEACHER_PY:-python}   # env able to serve Qwen/Qwen3.8-27B
PYPIPE=${PIPE_PY:-python}         # DCLM data-processing env (fastText LID)
PYDCLM=${DCLM_PY:-python}         # DCLM env with the baselines/ mappers (RefinedWeb-rule)
PYDT=${DATATROVE_PY:-python}      # env with datatrove==0.2.0 (FineWeb-rule)
PYJUDGE=${JUDGE_PY:-$PY}          # vLLM env able to serve openai/gpt-oss-120b

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false} VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
echo "[$(date '+%F %T %Z')] host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-none}"
