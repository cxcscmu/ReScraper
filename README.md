# ReScraper: Unified Scraping and Cleaning of Web Data for Effective LLM Pretraining

Code release for the ReScraper paper.

- Model: [`cx-cmu/ReScraper`](https://huggingface.co/cx-cmu/ReScraper)
- Data: [`cx-cmu/ReScraper-Data`](https://huggingface.co/datasets/cx-cmu/ReScraper-Data)

ReScraper replaces the heuristic HTML-to-text stack of pretraining pipelines (a rule-based scraper followed by
rule-based cleaning filters) with one small language model (Qwen3-0.6B). The model reads the visible text of a
raw web page, rendered one block per line with line ids `<lid:n>`, and generates a short program:

```
<keep>|<edit>|<delete>|<rewrite>      the operation chosen for the page
<extract>                             always first: remove boilerplate lines
rm A-B / rm A                         (lines that are not main content)
<same operation again>
payload                               <edit>: rm A-B / sub N: "s" (delete lines / strings); <rewrite>: new text
```

A deterministic executor (`lib/rescraper_ops.py`) applies the program to the rendered lines, so kept text is copied, not
regenerated. The supervision is built from three teachers run in sequence on the same rendering: Dripper
(main-content extraction -> `<extract>`), Qwen3.8-27B under a strict-subset refinement prompt (-> `<keep>`,
`<edit>`, `<delete>`), and the RePro 1B rephraser for teacher-deleted pages whose FineWeb-Edu score is at least 1.0
(-> `<rewrite>`). The student is fine-tuned in two stages (all operations; then a rewrite-heavy mixture), applied to
the raw HTML of the whole DCLM source pool, and the output is deduplicated and tokenized for DCLM pretraining at the
400M, 1B and 3B scales.

## Contents

- [Pipeline](#pipeline)
- [Setup](#setup)
- [Reproducing the released model and corpus](#reproducing-the-released-model-and-corpus)
- [License](#license)

## Pipeline

```
DCLM pool sample (raw HTML, ~10.3K shards, 18.0M pages)
 │
 ├─ data_construction/dripper/        Dripper over the pool (teacher 1; also the base of the scraper baselines)
 ├─ data_construction/seed_pages/     ~1.6M SFT seed pages (held-out shards excluded by stem)
 ├─ data_construction/teacher_refine/ Qwen3.8-27B refinement labels (teacher 2)
 ├─ data_construction/base_set/       render raw HTML -> <lid:n> input; serialized target; no-rewrite base set (1.38M)
 ├─ data_construction/rescue/         FineWeb-Edu of deleted pages; RePro 1B rewrites (teacher 3)
 ├─ data_construction/sft_sets/       Stage-1 set (1.38M) and Stage-2 set (131K, 30% <rewrite>)
 ├─ training/                         Stage 1 (Qwen3-0.6B, 3 epochs) -> Stage 2 (from the epoch-2 checkpoint, 3 epochs)
 ├─ inference/                        Stage-2 model over the pool (T=1.0, 3,072 new tokens) -> executor -> post-filter
 ├─ corpus/                           BFF dedup (13-gram, 0.8) -> GPT-NeoX-20B tokens (7.44B unique tokens)
 └─ pretraining/                      DCLM 411m_1x / 1b_1x_fast pretraining, 22-task Core evaluation (eval_sheet.py)

baselines/     rule stacks (C4 / RefinedWeb / FineWeb rules), scrapers (resiliparse, trafilatura, jusText, Dripper),
               ProX-C, UltraX
ablations/     operation ablation (Table 3) and the two-stage "Dripper <extract>" refiner
evaluation/    5,000-page held-out set, page-aligned outputs of every pipeline, teacher fidelity, extraction F1,
               gpt-oss-120b judges, DataMan / FineWeb-Edu scoring, n-gram diversity
analysis/      scripts behind every figure and table
prompts/       every prompt, verbatim
lib/           shared modules (renderer, line operations, executor, rephraser helpers)
```

## Setup

1. Environments (`requirements/README.md`):
   - refiner: Python 3.12, torch 2.9.0+cu128, transformers 4.57.6, vLLM 0.11.1, DeepSpeed, flash-attn 2.8.3, and
     `dripper==1.0.0` (MinerU-HTML, which provides the page renderer) -> `requirements/refiner.txt`;
   - teacher: an environment that can serve Qwen3.8-27B (vLLM 0.28.0, transformers 5.16.1) -> `requirements/teacher.txt`;
   - DCLM: upstream DCLM at commit `c59dd5878c8bd1965b4894e20d8a091e8fdeac7a` plus our patch -> `pretraining/README.md`,
     `requirements/dclm.txt`;
   - tools of the baselines and evaluation (datatrove 0.2.0, trafilatura 2.0.0, jusText 3.0.2, resiliparse 0.16.0,
     gpt-oss-120b via vLLM) -> READMEs of `baselines/` and `evaluation/`.
2. `cp configs/paths.env.example configs/paths.env`, edit, `source configs/paths.env`. All scripts read locations
   from these variables (`RESCRAPER_ROOT`, `WORK_DIR`, `DCLM_DIR`, model identifiers, python interpreters).
3. SLURM scripts are examples written for 8-GPU (H200) nodes. `#SBATCH --partition=gpu|cpu` are placeholders:
   override with `-p` or `SBATCH_PARTITION`; chains use `GPU_PARTITION` / `CPU_PARTITION`.

The source pool is a 10% shard sample of the DCLM `dclm-pool-400m-1x` pool (raw HTML, English pages); every pipeline
starts from the same pages. Held-out shards are listed in `$HELDOUT_SHARDS`.

## Reproducing the released model and corpus

The pipeline runs in seven stages. Each directory's README gives the exact commands, the inputs and outputs, and
the compute our own runs used.

1. **Sample the seed pages.** Draw the seed pages for supervision from the source pool
   (`data_construction/seed_pages/`).
2. **Extract the main content — teacher 1.** Run Dripper over the pool; its main-content decision becomes the
   `<extract>` line removals (`data_construction/dripper/`).
3. **Label the operation — teacher 2.** Qwen3.8-27B, under a strict-subset refinement prompt, assigns `<keep>`,
   `<edit>` or `<delete>` and produces the edits (`data_construction/teacher_refine/`). `data_construction/base_set/`
   joins the two teachers on the same rendering into the base training set.
4. **Rescue deleted pages — teacher 3.** Score the teacher-deleted pages with FineWeb-Edu; those at 1.0 or above are
   rephrased by RePro 1B and relabelled `<rewrite>` (`data_construction/rescue/`).
5. **Train the refiner.** Two supervised stages: first all operations, then a rewrite-heavy mixture so the model
   learns the operation it would otherwise almost never see (`training/`).
6. **Refine the whole pool.** Apply the trained model to the raw HTML of every source page, then run the post-filter,
   Bloom-filter deduplication and tokenization (`inference/`, `corpus/`).
7. **Pretrain and evaluate.** Train 400M, 1B and 3B models on the resulting corpus and score them on DCLM Core
   (`pretraining/`).

## License

Apache-2.0 (`LICENSE`). Third-party models and tools keep their own licenses.
