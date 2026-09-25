#!/usr/bin/env python3
"""Extended operation statistics of the released model's programs on the 5,000 held-out pages (heldout5k).
Table 11 (tab:operation-stats); --tex writes the tabular.

Programs: $WORK_DIR/eval/heldout_pipeline/work/heldout5k_rel/ours.jsonl `raw` (release decoding, T=1.0,
top_p=1.0, max_tokens=3072); the 11 pages the release run never sent to the model have no program and are
left out (4,989 programs). The teacher programs of the same pages ($WORK_DIR/eval/heldout5k/heldout5k.jsonl
`output`, same serialization) are measured with the same code as a reference.

Program grammar (decision-first staged format, lib/rescraper_ops.py): line 1 = decision; "<extract>";
the extractor's removals; the repeated decision; the payload (edit ops for <edit>, text for <rewrite>).
Ops: "rm A" / "rm A-B" remove whole input lines (1-based, inclusive); "sub N: \"s\"" removes substring s
from line N. Lines = the non-empty lines of the numbered input, as the executor (editops.apply_ops) counts them.

Measured
  <extract> removals : every page with a program (all decisions). rm ops per page, removed lines per page,
                       span length of each rm op in lines and in words (whitespace words of the removed lines),
                       position of each removed line in the page (first / middle / last third by line index),
                       and whether it lies in the page's leading run (a removed block starting at line 1),
                       trailing run (a removed block ending at the last line) or in the interior.
  <edit> removals    : pages whose decision is <edit> and whose payload parses as ops (as the executor reads it;
                       payloads that are page text are counted separately as "full-text fallback").
                       rm ops and sub ops per page; rm span length in lines / words (only lines not already
                       removed by <extract> count toward words; rm lines already removed are reported);
                       sub span length in words and characters; position of each edit (removed line or sub
                       line) within the text left after <extract> (thirds by rank among the kept lines), and
                       whether it touches the first / last kept line.
usage: analyze_operation_stats_5k.py [--out $FIG_DATA_DIR/operation_stats_5k.json] [--tex FILE]
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import rescraper_ops as X  # noqa: E402
import editops as E  # noqa: E402

HERE = Path(__file__).resolve().parent  # analysis/
DATA_DIR = Path(os.environ.get("FIG_DATA_DIR", HERE / "data"))
WORK_DIR = os.environ.get("WORK_DIR", "")  # see configs/paths.env.example
GOLD = os.path.join(WORK_DIR, "eval", "heldout5k", "heldout5k.jsonl")
OURS = os.path.join(WORK_DIR, "eval", "heldout_pipeline", "work", "heldout5k_rel", "ours.jsonl")
WORD = re.compile(r"\S+")
THIRDS = ("first", "middle", "last")


def third(rank0, n):
    """rank0 in [0, n) -> 0/1/2 by the position of the line's centre."""
    return min(2, int(3 * (rank0 + 0.5) / n))


def dist(xs):
    xs = sorted(xs)
    if not xs:
        return {"n": 0}
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "mean": statistics.fmean(xs), "p25": q(0.25), "median": statistics.median(xs),
            "p75": q(0.75), "p90": q(0.90), "max": xs[-1], "sum": sum(xs)}


def parse_ops(block):
    rms, subs, other = [], [], []
    for l in block.split("\n"):
        s = l.strip()
        if not s:
            continue
        m = E.OPS_RM.match(s)
        if m:
            a = int(m.group(1)); b = int(m.group(2) or a); rms.append((a, b)); continue
        m = E.OPS_SUB.match(s)
        if m:
            try:
                sub = json.loads(m.group(2))
            except Exception:
                sub = m.group(2).strip("\"")
            subs.append((int(m.group(1)), sub)); continue
        other.append(s)
    return rms, subs, other


def analyze(pages, programs):
    """pages: list of input strings; programs: list of raw program strings (same order)."""
    A = collections.defaultdict(list)          # distributions
    C = collections.Counter()                  # counts
    pos_ext_lines = [0, 0, 0]; pos_ext_words = [0, 0, 0]; base_lines = [0, 0, 0]; base_words = [0, 0, 0]
    run_lines = collections.Counter(); run_words = collections.Counter()
    pos_edit = collections.Counter(); pos_edit_words = collections.Counter()
    hist_ext = [0] * 10; hist_edit = [0] * 10
    for inp, raw in zip(pages, programs):
        src = [E.LID.sub("", x) for x in E.lines_of(inp)]
        n = len(src)
        wl = [len(WORD.findall(x)) for x in src]
        lines = raw.strip().split("\n")
        d = lines[0].strip() if lines else ""
        if d == "<extract>":
            # staged (non-decision-first) serialization, as in the teacher programs of heldout5k:
            # <extract> block first, then the decision and its payload
            ops1, tag2, rest = X.parse_staged(raw)
            d = tag2
            C["staged_format"] += 1
        else:
            ops1, tag2, rest = X.parse_staged("\n".join(lines[1:]))
        if d not in X.DFIRST_TAGS:
            C["unparseable_decision"] += 1
            continue
        C["programs"] += 1; C["dec_" + d] += 1
        rms, subs, _ = parse_ops(ops1)
        # ---- page baseline by thirds
        for i in range(n):
            base_lines[third(i, n)] += 1; base_words[third(i, n)] += wl[i]
        # ---- <extract>
        ext = set()
        for a, b in rms:
            span = [i for i in range(a, b + 1) if 1 <= i <= n]
            A["ext_span_lines"].append(len(span))
            A["ext_span_words"].append(sum(wl[i - 1] for i in span))
            if len(span) < b - a + 1:
                C["ext_rm_lines_out_of_range"] += (b - a + 1) - len(span)
            ext.update(span)
        C["ext_sub_ops"] += len(subs)
        A["ext_rm_ops_per_page"].append(len(rms))
        A["ext_removed_lines_per_page"].append(len(ext))
        A["ext_removed_line_share_per_page"].append(len(ext) / n if n else 0.0)
        C["ext_pages_with_removal"] += bool(ext)
        C["ext_pages_all_lines_removed"] += bool(n) and len(ext) == n
        lead = 0
        while lead + 1 in ext:
            lead += 1
        trail = 0
        while n - trail in ext and n - trail > lead:
            trail += 1
        C["ext_pages_head_removed"] += lead > 0
        C["ext_pages_tail_removed"] += trail > 0
        C["ext_pages_head_and_tail_removed"] += lead > 0 and trail > 0
        for i in ext:
            t = third(i - 1, n)
            pos_ext_lines[t] += 1; pos_ext_words[t] += wl[i - 1]
            hist_ext[min(9, int(10 * (i - 0.5) / n))] += 1
            r = "leading" if i <= lead else "trailing" if i > n - trail else "interior"
            run_lines[r] += 1; run_words[r] += wl[i - 1]
        # ---- <edit>
        if d != "<edit>":
            continue
        C["edit_pages"] += 1
        if tag2 not in X.DFIRST_TAGS:
            rest = (tag2 + "\n" + rest) if tag2 else rest
        bl = [l.strip() for l in rest.split("\n") if l.strip()]
        if not bl:
            C["edit_empty_payload"] += 1
            continue
        if not all(E.OPS_RM.match(l) or E.OPS_SUB.match(l) for l in bl):
            C["edit_fulltext_fallback"] += 1
            continue
        C["edit_ops_pages"] += 1
        erms, esubs, _ = parse_ops("\n".join(bl))
        A["edit_rm_ops_per_page"].append(len(erms)); A["edit_sub_ops_per_page"].append(len(esubs))
        A["edit_ops_per_page"].append(len(erms) + len(esubs))
        C["edit_mix_" + ("rm+sub" if erms and esubs else "rm_only" if erms else "sub_only" if esubs else "none")] += 1
        kept = [i for i in range(1, n + 1) if i not in ext]
        rank = {i: k for k, i in enumerate(kept)}
        touched_first = touched_last = False
        edit_lines = set()
        for a, b in erms:
            span = [i for i in range(a, b + 1) if 1 <= i <= n]
            new = [i for i in span if i not in ext]
            C["edit_rm_lines_already_extracted"] += len(span) - len(new)
            A["edit_rm_span_lines"].append(len(span))
            A["edit_rm_span_words"].append(sum(wl[i - 1] for i in new))
            edit_lines.update(new)
        for i in edit_lines:
            k = rank[i]
            pos_edit[("rm", third(k, len(kept)))] += 1; pos_edit_words[("rm", third(k, len(kept)))] += wl[i - 1]
            hist_edit[min(9, int(10 * (k + 0.5) / len(kept)))] += 1
            touched_first |= k == 0; touched_last |= k == len(kept) - 1
        for ln, s in esubs:
            A["edit_sub_span_words"].append(len(WORD.findall(s))); A["edit_sub_span_chars"].append(len(s))
            if not (1 <= ln <= n) or ln in ext:
                C["edit_sub_on_removed_or_missing_line"] += 1
                continue
            C["edit_sub_found_in_line"] += s in src[ln - 1]
            k = rank[ln]
            pos_edit[("sub", third(k, len(kept)))] += 1; pos_edit_words[("sub", third(k, len(kept)))] += len(WORD.findall(s))
            hist_edit[min(9, int(10 * (k + 0.5) / len(kept)))] += 1
            touched_first |= k == 0; touched_last |= k == len(kept) - 1
            # sub at the start / end of its line?
            x = src[ln - 1]; j = x.find(s)
            if j >= 0:
                C["edit_sub_at_line_start"] += x[:j].strip() == ""
                C["edit_sub_at_line_end"] += x[j + len(s):].strip() == ""
        C["edit_pages_touching_first_kept_line"] += touched_first
        C["edit_pages_touching_last_kept_line"] += touched_last
    share = lambda v: [x / max(1, sum(v)) for x in v]
    rs = sum(run_lines.values()); rw = sum(run_words.values())
    out = {
        "counts": dict(C),
        "distributions": {k: dist(v) for k, v in A.items()},
        "extract_position": {
            "removed_lines_by_third": dict(zip(THIRDS, share(pos_ext_lines))),
            "removed_words_by_third": dict(zip(THIRDS, share(pos_ext_words))),
            "page_lines_by_third": dict(zip(THIRDS, share(base_lines))),
            "page_words_by_third": dict(zip(THIRDS, share(base_words))),
            "removed_lines_by_run": {k: run_lines[k] / max(1, rs) for k in ("leading", "interior", "trailing")},
            "removed_words_by_run": {k: run_words[k] / max(1, rw) for k in ("leading", "interior", "trailing")},
            "removed_lines_decile_hist": hist_ext,
        },
        "edit_position_in_extracted_text": {
            "rm_lines_by_third": dict(zip(THIRDS, share([pos_edit[("rm", t)] for t in range(3)]))),
            "sub_ops_by_third": dict(zip(THIRDS, share([pos_edit[("sub", t)] for t in range(3)]))),
            "all_edits_by_third": dict(zip(THIRDS, share([pos_edit[("rm", t)] + pos_edit[("sub", t)] for t in range(3)]))),
            "rm_words_by_third": dict(zip(THIRDS, share([pos_edit_words[("rm", t)] for t in range(3)]))),
            "edits_decile_hist": hist_edit,
        },
    }
    return out


def fmt_iqr(d, f="%g"):
    return (f + " [" + f + "--" + f + "]") % (d["median"], d["p25"], d["p75"])


def tex(res):
    S, T = res["student"], res["teacher"]
    pc = lambda x: "%.1f" % (100 * x)
    def row(label, fn):
        return "  %s & %s & %s \\\\" % (label, fn(S), fn(T))
    D = lambda r, k: r["distributions"][k]
    C = lambda r, k: r["counts"].get(k, 0)
    L = [
        "% Generated by analysis/analyze_operation_stats_5k.py -> operation_stats_5k.json",
        "% Medians with [p25--p75] (mean in parentheses); words = whitespace words; teacher = the SFT-rule labels of the same pages.",
        "\\begin{tabular}{lrr}",
        "  \\toprule",
        "  & \\textbf{\\ours} & \\textbf{Teacher} \\\\",
        "  \\midrule",
        "  \\multicolumn{3}{l}{\\textit{\\texttt{<extract>} removals, all pages}} \\\\",
        row("Pages", lambda r: "{:,}".format(C(r, "programs"))),
        row("Pages with a removal (\\%)", lambda r: pc(C(r, "ext_pages_with_removal") / C(r, "programs"))),
        row("\\texttt{rm} ops per page", lambda r: fmt_iqr(D(r, "ext_rm_ops_per_page")) + " (%.1f)" % D(r, "ext_rm_ops_per_page")["mean"]),
        row("Lines removed per page", lambda r: fmt_iqr(D(r, "ext_removed_lines_per_page"))),
        row("Span length, lines", lambda r: fmt_iqr(D(r, "ext_span_lines"))),
        row("Span length, words", lambda r: fmt_iqr(D(r, "ext_span_words"))),
        row("Removed lines: first / middle / last third (\\%)", lambda r: "/".join(
            "%.0f" % (100 * r["extract_position"]["removed_lines_by_third"][k]) for k in THIRDS)),
        row("Pages with a leading / trailing removed block (\\%)", lambda r: "%.0f/%.0f" % (
            100 * C(r, "ext_pages_head_removed") / C(r, "programs"), 100 * C(r, "ext_pages_tail_removed") / C(r, "programs"))),
        row("Removed words: leading / interior / trailing (\\%)", lambda r: "/".join(
            "%.0f" % (100 * r["extract_position"]["removed_words_by_run"][k]) for k in ("leading", "interior", "trailing"))),
        "  \\midrule",
        "  \\multicolumn{3}{l}{\\textit{\\texttt{<edit>} removals, edited pages}} \\\\",
        row("Edited pages (ops / page text)", lambda r: "{:,} / {:,}".format(C(r, "edit_ops_pages"), C(r, "edit_fulltext_fallback"))),
        row("\\texttt{rm} ops per page", lambda r: fmt_iqr(D(r, "edit_rm_ops_per_page")) + " (%.1f)" % D(r, "edit_rm_ops_per_page")["mean"]),
        row("\\texttt{sub} ops per page", lambda r: fmt_iqr(D(r, "edit_sub_ops_per_page")) + " (%.1f)" % D(r, "edit_sub_ops_per_page")["mean"]),
        row("\\texttt{rm} only / \\texttt{sub} only / both (\\%)", lambda r: "/".join(
            "%.0f" % (100 * C(r, "edit_mix_" + k) / max(1, C(r, "edit_ops_pages"))) for k in ("rm_only", "sub_only", "rm+sub"))),
        row("\\texttt{rm} span, lines", lambda r: fmt_iqr(D(r, "edit_rm_span_lines"))),
        row("\\texttt{rm} span, words", lambda r: fmt_iqr(D(r, "edit_rm_span_words"))),
        row("\\texttt{sub} span, words", lambda r: fmt_iqr(D(r, "edit_sub_span_words"))),
        row("Edits: first / middle / last third (\\%)", lambda r: "/".join(
            "%.0f" % (100 * r["edit_position_in_extracted_text"]["all_edits_by_third"][k]) for k in THIRDS)),
        row("Pages editing the first / last kept line (\\%)", lambda r: "%.0f/%.0f" % (
            100 * C(r, "edit_pages_touching_first_kept_line") / max(1, C(r, "edit_ops_pages")),
            100 * C(r, "edit_pages_touching_last_kept_line") / max(1, C(r, "edit_ops_pages")))),
        "  \\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA_DIR / "operation_stats_5k.json")
    ap.add_argument("--tex", type=Path, default=None)
    args = ap.parse_args()
    if not WORK_DIR:
        sys.exit("set WORK_DIR (see configs/paths.env.example)")
    gold = [json.loads(l) for l in open(GOLD)]
    ours = [json.loads(l) for l in open(OURS)]
    assert [g["gid"] for g in gold] == list(range(len(gold))) == [o["gid"] for o in ours]
    ok = [(g, o) for g, o in zip(gold, ours) if o["status"] == "ok"]
    res = {
        "description": "Operation statistics of the released ReScraper programs (release decoding T=1.0, top_p=1.0, "
                       "max_tokens=3072) on the %d of 5,000 held-out pages (heldout5k) the release run processed; the "
                       "teacher programs of the same pages are measured with the same code. See the script docstring "
                       "for definitions." % len(ok),
        "source": OURS + " (raw) ; " + GOLD + " (input, output)",
        "skipped_not_sent": [o["gid"] for o in ours if o["status"] != "ok"],
        "student": analyze([g["input"] for g, _ in ok], [o["raw"] for _, o in ok]),
        "teacher": analyze([g["input"] for g, _ in ok], [g["output"] for g, _ in ok]),
    }
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    if args.tex:
        args.tex.write_text(tex(res))
    for who in ("student", "teacher"):
        r = res[who]
        print("==", who, json.dumps(r["counts"]))
        for k, v in r["distributions"].items():
            print("   %-34s n=%-7d mean=%-8.2f p25=%-6g med=%-6g p75=%-6g p90=%-6g max=%g" % (
                k, v["n"], v["mean"], v["p25"], v["median"], v["p75"], v["p90"], v["max"]))
        print("   extract_position", json.dumps({k: (v if isinstance(v, list) else {a: round(b, 3) for a, b in v.items()}) for k, v in r["extract_position"].items()}))
        print("   edit_position", json.dumps({k: (v if isinstance(v, list) else {a: round(b, 3) for a, b in v.items()}) for k, v in r["edit_position_in_extracted_text"].items()}))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
