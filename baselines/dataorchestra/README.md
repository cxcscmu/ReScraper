# baselines/dataorchestra/

DataOrchestra (Huang et al., 2026) is the multi-agent baseline in the main table (row marked `*`, 13.60B unique
tokens). Its orchestrator is Qwen3-1.7B, with a fine-tuned Qwen3-0.6B for line removal and Qwen3-4B for
instruction-guided rewriting. The DataOrchestra corpus was produced with the authors' released system over the
scraped text of our source pool. Our code for running it is not included in this repository. Pretraining and
evaluation on the corpus use `pretraining/`, like every other corpus.
