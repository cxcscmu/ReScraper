#!/bin/bash
# Directory table for every corpus that goes through the shared finishing chain
# (post-filter -> BFF dedup -> GPT-NeoX-20B tokenization -> DCLM pretraining).
# Source it and call `corpus_vars <name>`; it sets:
#   CDIR      corpus working directory
#   READ      DCLM readable dataset name (exp_data/datasets/tokenized/<READ>.json, the pretraining --data-config)
#   DEDUP_IN  input of the BFF dedup (post-filtered text for model outputs)
#   DEDUP_OUT, TOK_IN, TOK_OUT, MARKD, LOGD, PTL (pretraining logs)
# plus CODE (the DCLM checkout, $DCLM_DIR), PY (python of the DCLM data-processing env, $PIPE_PY) and R ($WORK_DIR).
# No rule filter and no fastText filter are applied to any of these corpora ("norule_noft").
R=${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
CODE=${DCLM_DIR:?set DCLM_DIR}
PY=${PIPE_PY:-python}

corpus_vars(){ # $1 = corpus name
  CORPUS=$1
  case $1 in
    rescraper)                     # the released ReScraper corpus (Stage-2 model over the pool, T=1)
      CDIR=$R/dclm_pipeline/rescraper_corpus
      READ=rescraper_norule_noft
      DEDUP_IN=$CDIR/text_clean ;;
    ablation_full|ablation_nodel|ablation_noedit|ablation_noext|ablation_norwdel|ablation_norw)
      # operation ablation arms, all built from one inference pass (ablations/operation_ablation/)
      CDIR=$R/dclm_pipeline/operation_ablation/${1#ablation_}
      READ=${1}_norule_noft
      DEDUP_IN=$CDIR/text_clean ;;
    two_stage_refiner)             # Dripper extraction -> text-to-text refiner (ablations/two_stage_refiner/)
      CDIR=$R/dclm_pipeline/two_stage_pool
      READ=two_stage_refiner_norule_noft
      DEDUP_IN=$CDIR/best ;;
    prox_c)                        # resiliparse extraction -> ProX-C (baselines/prox_c/)
      CDIR=$R/dclm_pipeline/prox_c
      READ=prox_c_norule_noft
      DEDUP_IN=$CDIR/text ;;
    ultrax)                        # resiliparse extraction (no filter) -> UltraX (baselines/ultrax/)
      CDIR=$R/dclm_pipeline/resiliparse_raw_ultrax
      READ=resiliparse_raw_ultrax_dedup_noft
      DEDUP_IN=$CDIR/text ;;
    refinedweb_rule)               # resiliparse -> RefinedWeb rules (Table 2 RefinedWeb-rule; Fig. 3 resiliparse bar)
      CDIR=$R/dclm_pipeline/resiliparse
      READ=resiliparse_no_fasttext
      DEDUP_IN=$CDIR/step3b_post_lang/dclm_baseline_refinedweb_post_lang/processed_data ;;
    dripper_refinedweb_rule)       # Dripper -> RefinedWeb rules (Fig. 3 Dripper bar, 1B)
      CDIR=$R/dclm_pipeline/dripper_refinedweb_rule
      READ=dripper_refinedweb_rule_noft
      DEDUP_IN=$R/dclm_pipeline/dripper/step3b_post_lang/dclm_baseline_refinedweb_post_lang/processed_data ;;
    dripper_norule)                # Dripper text without a rule stack (Fig. 3 Dripper bar, 400M)
      CDIR=$R/dclm_pipeline/dripper_norule
      READ=dripper_norule_noft
      DEDUP_IN=${DRIPPER_STEP2_DIR:-$R/dclm_pipeline/dripper/step2_text} ;;
    dripper_ultrax)                # Dripper -> UltraX (Fig. 3 Dripper + UltraX bar)
      CDIR=$R/dclm_pipeline/dripper_ultrax
      READ=dripper_ultrax_noft
      DEDUP_IN=$CDIR/text ;;
    ultrax_rwrule)                 # UltraX over the RefinedWeb-rule corpus (Table 2 UltraX, 3B row); NOT deduplicated again
      CDIR=$R/dclm_pipeline/resiliparse_ultrax
      READ=resiliparse_ultrax_norule_noft
      DEDUP_IN=$CDIR/text ;;
    *) echo "unknown corpus $1" >&2; return 2 ;;
  esac
  DEDUP_OUT=$CDIR/step4_dedup_oldboth_65b
  TOK_IN=$DEDUP_OUT
  # ultrax_rwrule cleaned an already-deduplicated, already-rule-filtered corpus: tokenize its text directly
  [[ $1 == ultrax_rwrule ]] && TOK_IN=$CDIR/text
  TOK_OUT=$CDIR/step6_tokenized_no_fasttext
  PTL=$CDIR/pretrain_logs
  MARKD=$CDIR/.marks
  LOGD=$CDIR/logs_pipeline
}
