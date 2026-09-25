#!/usr/bin/env python3
"""Token volume of the held-out pages after each cumulative stage of ReScraper (GPT-NeoX-20B tokens).
Data of Figure 9 (fig:token-waterfall), plotted by plot_token_waterfall_5k.py.

Stage definitions, per page with decision d = line 1 of the raw program:
  input   = tokens of the rendering the model reads, <lid:N> markers stripped
  extract = tokens of apply_ops(input, stage-1 <extract> ops)            (every page)
  delete  = 0 if d == <delete> else extract
  edit    = 0 if <delete>; final text if d == <edit>; else extract
  rewrite = 0 if <delete>; final text if d in (<edit>, <rewrite>); else extract   (= the page's corpus text)
Default input: the 5,000 held-out pages ($WORK_DIR/eval/heldout5k/heldout5k.jsonl) x the release-decoding
outputs ($WORK_DIR/eval/heldout_pipeline/work/heldout5k_rel/ours.jsonl, written by evaluation/heldout_pipeline/).
The 11 pages the release run never sent to the model (prompt too long) have no program; they are excluded
from the stage totals (reported separately under "skipped_not_sent").
--dump <file.jsonl> instead runs on a JSONL of {inp, raw, pfinal} records (model input, raw program, executed
text), e.g. a raw-output dump of a held-out evaluation run.
The tokenizer is loaded from the Hugging Face cache (set HF_HOME).
usage: compute_token_waterfall_5k.py [--out $FIG_DATA_DIR/token_waterfall_5k.json] [--limit N] [--dump FILE]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import editops as E  # noqa: E402
from rescraper_ops import parse_staged  # noqa: E402

HERE = Path(__file__).resolve().parent  # analysis/
DATA_DIR = Path(os.environ.get("FIG_DATA_DIR", HERE / "data"))
WORK_DIR = os.environ.get("WORK_DIR", "")  # see configs/paths.env.example
GOLD = os.path.join(WORK_DIR, "eval", "heldout5k", "heldout5k.jsonl")
OURS = os.path.join(WORK_DIR, "eval", "heldout_pipeline", "work", "heldout5k_rel", "ours.jsonl")
LID = re.compile(r"^<lid:\d+> ?", re.M)
STAGES = ["input", "extract", "delete", "edit", "rewrite"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA_DIR / "token_waterfall_5k.json")
    ap.add_argument("--limit", type=int, default=None, help="first N pages only (e.g. 960)")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    def T(texts):
        texts = [t or "" for t in texts]
        ids = tok(texts, add_special_tokens=False)["input_ids"]
        return [len(x) if t else 0 for x, t in zip(ids, texts)]

    recs, skipped = [], []
    if args.dump:
        for line in open(args.dump):
            d = json.loads(line)
            recs.append({"gid": len(recs), "inp": d["inp"], "raw": d["raw"], "final": d["pfinal"] or ""})
        source = args.dump
    else:
        if not WORK_DIR:
            sys.exit("set WORK_DIR (see configs/paths.env.example)")
        gold = [json.loads(l) for l in open(GOLD)]
        ours = [json.loads(l) for l in open(OURS)]
        assert [g["gid"] for g in gold] == list(range(len(gold))) == [o["gid"] for o in ours]
        for g, o in zip(gold, ours):
            if o["status"] != "ok":
                skipped.append({"gid": g["gid"], "status": o["status"], "inp": g["input"]})
                continue
            recs.append({"gid": g["gid"], "inp": g["input"], "raw": o["raw"],
                         "final": "" if o["pt"] == "<delete>" else (o["text"] or ""), "pt": o["pt"]})
        source = GOLD + " x " + OURS
    if args.limit:
        recs = [r for r in recs if r["gid"] < args.limit]
        skipped = [r for r in skipped if r["gid"] < args.limit]

    decs, t_in_s, t_ex_s, t_af_s = [], [], [], []
    for r in recs:
        lines = r["raw"].strip().split("\n")
        decs.append(lines[0].strip())
        ops1, _, _ = parse_staged("\n".join(lines[1:]))
        t_in_s.append(LID.sub("", r["inp"]).strip())
        t_ex_s.append(E.apply_ops(r["inp"], ops1).strip())
        t_af_s.append((r["final"] or "").strip())
    t_in, t_ex, t_af = T(t_in_s), T(t_ex_s), T(t_af_s)
    st = {k: 0 for k in STAGES}
    by_dec = {}
    final_chk = 0
    for dec, a, b, c in zip(decs, t_in, t_ex, t_af):
        st["input"] += a
        st["extract"] += b
        st["delete"] += 0 if dec == "<delete>" else b
        st["edit"] += 0 if dec == "<delete>" else (c if dec == "<edit>" else b)
        st["rewrite"] += 0 if dec == "<delete>" else (c if dec in ("<edit>", "<rewrite>") else b)
        final_chk += c
        s = by_dec.setdefault(dec, {"pages": 0, "input": 0, "extract": 0, "final": 0})
        s["pages"] += 1; s["input"] += a; s["extract"] += b; s["final"] += c
    assert st["rewrite"] == final_chk, (st["rewrite"], final_chk)
    sk_tokens = T([LID.sub("", r["inp"]).strip() for r in skipped]) if skipped else []
    out = {
        "pages": len(recs),
        "tokenizer": "EleutherAI/gpt-neox-20b",
        "tokens": st,
        "share_of_input": {k: v / st["input"] for k, v in st.items()},
        "by_decision": by_dec,
        "skipped_not_sent": {"pages": len(skipped), "gids": [r["gid"] for r in skipped],
                             "input_tokens": sum(sk_tokens),
                             "note": "never sent to the model by the release run (prompt too long); excluded above"},
        "description": "GPT-NeoX-20B tokens of the %d held-out pages the released model processed (release decoding: "
                       "T=1.0, top_p=1.0, max_tokens=3072) after each cumulative stage of ReScraper (input rendering, "
                       "extract, delete, edit, rewrite)." % len(recs),
        "source": source,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for k, v in st.items():
        print("%-8s %10d  %6.2f%%" % (k, v, 100 * v / st["input"]))
    print("skipped:", out["skipped_not_sent"]["pages"], "pages,", out["skipped_not_sent"]["input_tokens"], "input tokens")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
