"""Default locations of every input and output used under evaluation/.

Everything lives under $WORK_DIR (required; see configs/paths.env.example); each location can be overridden with its
own environment variable. The shell counterpart, sourced by the SLURM scripts, is evaluation/env.sh.
"""
import os

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LIB_DIR = os.path.join(REPO_DIR, "lib")
PROMPTS_DIR = os.path.join(REPO_DIR, "prompts")

WORK_DIR = os.environ.get("WORK_DIR", "")
if not WORK_DIR:
    raise SystemExit("WORK_DIR is not set (see configs/paths.env.example)")


def _p(var, *default):
    return os.environ.get(var) or os.path.join(WORK_DIR, *default)


# ---- outputs of evaluation/
EVAL_DIR = _p("EVAL_DIR", "eval")
HELDOUT5K_DIR = os.environ.get("HELDOUT5K_DIR") or os.path.join(EVAL_DIR, "heldout5k")      # heldout5k.jsonl, html5k.jsonl
HELDOUT960_DIR = os.environ.get("HELDOUT960_DIR") or os.path.join(EVAL_DIR, "heldout960")   # edu1 relabel of rows 0..959
HELDOUT_PIPELINE_DIR = os.environ.get("HELDOUT_PIPELINE_DIR") or os.path.join(EVAL_DIR, "heldout_pipeline")

# ---- source pool (DCLM 400M-1x pool, 10% raw-HTML sample) and the pipelines run over it
POOL_RAW_DIR = _p("HTML_POOL_DIR", "pools", "dclm-pool-400m-1x-html-jsonl-step3a-10pct")
POOL_SIMP_DIR = _p("POOL_SIMP_DIR", "pools", "dclm-pool-400m-1x-simp-html-jsonl-step3a-10pct", "step3b_length_only",
                   "dclm_baseline_refinedweb_post_lang_length_only")      # simplified-html rows of the earlier held-out table
RESILIPARSE_EXTRACT_DIR = _p("RESILIPARSE_EXTRACT_DIR", "dclm_pipeline", "resiliparse", "resiliparse_extract",
                             "resiliparse_extract", "processed_data")
TWO_STAGE_POOL_DIR = _p("TWO_STAGE_POOL_DIR", "dclm_pipeline", "two_stage_pool")          # {input,best,edu,rwrepro_olmo_temp1}/<stem>
TEXT_DECISIONS_DIR = _p("TEXT_DECISIONS_DIR", "dclm_pipeline", "rescue_add", "text_decisions")
ULTRAX_RERUN_DIR = _p("ULTRAX_RERUN_DIR", "dclm_pipeline", "resiliparse_raw_ultrax")       # {inf,post}/<stem>_processed.parquet
ABLATION_DIR = _p("ABLATION_DIR", "dclm_pipeline", "operation_ablation")                  # raw/, full/text, full/text_clean

# ---- SFT data (held-out exclusion lists and the earlier 3,161-page staged held-out table)
SFT_DIR = _p("SFT_DIR", "sft_data")
SFT_DATA_DIR = os.environ.get("SFT_DATA_DIR") or SFT_DIR   # dripper_sft_pool{,2}_{train,heldout}.jsonl
HELDOUT3161_DIR = os.environ.get("HELDOUT3161_DIR") or os.path.join(SFT_DATA_DIR, "heldout3161")
STAGE1_SFT_STEMS = os.environ.get("STAGE1_SFT_STEMS") or os.path.join(SFT_DIR, "stems_1p5m.txt")
TWO_STAGE_SFT_SOURCE_GLOB = os.environ.get("TWO_STAGE_SFT_SOURCE_GLOB") or os.path.join(SFT_DIR, "step1_output", "input_shard_*.jsonl")

# the 960 validation rows all come from this public Common Crawl shard of the pool
HELDOUT960_STEM = os.environ.get("HELDOUT960_STEM", "CC-MAIN-20140707234035-00041-ip-10-180-212-248.ec2.internal")
