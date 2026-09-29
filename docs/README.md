# Project page

The source of <https://cxcscmu.github.io/ReScraper/> (GitHub Pages, branch `main`, folder `/docs`).

Static project page (like `jerryyan123.github.io/TraceML`): a short summary, the paper figures, and
an interactive viewer over 39 held-out pages showing what each pipeline kept — RefinedWeb-rule,
FineWeb-rule, ProX-C, UltraX and ReScraper, all on the same source page.

```
index.html            the page
assets/style.css      tokens + layout (light/dark)
assets/app.js         the case viewer
assets/cases.js       window.CASES — the 39 cases (663 KB)
assets/figs/*.png     figures from ../ReScraper-ICLR-2027/figures/*.pdf, rendered with
                      `gs -sDEVICE=png16m -r<dpi>` at whatever dpi puts them at ~2600 px wide
                      (small PDFs need 600-930 dpi; anything less looks soft on a retina screen)
tools/                scripts that rebuild assets/cases.js
```

## Hosting

This folder is the only copy of the page. GitHub Pages serves it as is (branch `main`, folder
`/docs`, no build step), so editing files here and pushing to `main` republishes
<https://cxcscmu.github.io/ReScraper/> within a minute or two. Every path inside `index.html` is
relative, so the folder also works under any other prefix.

## Links the page points at

| | |
|---|---|
| Paper | <https://arxiv.org/abs/2609.34287> |
| Code | <https://github.com/cxcscmu/ReScraper> |
| Model | <https://huggingface.co/cx-cmu/ReScraper> |
| Data | <https://huggingface.co/datasets/cx-cmu/ReScraper-Data> |

The review-time anonymous mirror is **not** referenced anywhere on the page and must not be touched.

## Rebuilding the cases

Runs on the machine that holds the held-out page table and every system's per-page output
(paths in the table below are the authors' working copies, not part of this repository):

```bash
cd docs/tools
python3 build_pool.py pool.jsonl     # joins gold5k + all five systems + judge + scores
python3 select.py                    # ranks candidates per operation
python3 select.py show rewrite 5     # inspect candidates before picking
python3 build_cases.py               # writes ../assets/cases.js
```

Inputs (all read-only):

| what | path |
|---|---|
| held-out page table | `~/judge_babel/gold5k/gold5k.jsonl` |
| ReScraper outputs (release decoding) | `~/judge_babel/sec5_5k/work/gold5k_rel/ours.jsonl` |
| baselines | `~/judge_babel/sec5_5k/work/gold5k/sys_{ultrax,proxc,refinedweb_rule,fineweb_rule}.jsonl` |
| judge verdicts | `~/judge_babel/stu_teacher_why/runs/keep_judge_5000.jsonl` |
| DataMan / FineWeb-Edu per system | `~/ReScraper-ICLR-2027/data/quality_buckets.json` |
| rewrite faithfulness | `~/judge_babel/appendix_0925/faithfulness/runs/metrics.jsonl` |

The case list is hand-picked in `build_cases.py` (`KEEP` / `EDIT` / `DELETE` / `REWRITE` gid lists).
Rewrite cases are gated three ways before they go on the page: the judge must call the page worth
keeping, BERTScore-F1 against ReScraper's own extraction must be ≥ 0.92, and the faithfulness run
must find **no** entity and **no** number in the rewrite that is absent from the page
(`ent_novel_page_n == 0 and num_novel_page_n == 0`). Each one shown was then read by hand.
