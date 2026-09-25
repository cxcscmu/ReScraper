"""Compose the serialized two-stage target for every joined seed page (lib/rescraper_ops.to_staged_row):
    <extract>\n{rm a-b lines Dripper removed}\n<keep> | <edit>\n{rm/sub ops} | <delete> | <rewrite>\n{text}
Stage 1 (<extract>) = the lines of the rendered page that Dripper dropped (rows whose Dripper text is not an exact
line subset of the rendering are dropped: stage1_not_subset / stage1_has_sub / stage1_len_mismatch). Stage 2 =
the refining teacher's decision on the same line ids (a refined text that is not an exact subset edit is kept as
literal text: edit_fulltext_fallback; an empty diff becomes <keep>: refine_noop_to_keep).
usage: build_staged.py <dir with join_*.jsonl> <out.jsonl>
"""
import json, glob, collections, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import rescraper_ops as X

D, dst = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for f in sorted(glob.glob(D + "/join_*.jsonl")) for l in open(f, encoding="utf-8")]
n = len(rows)
print("joined rows", n)
C = collections.Counter(); fc = open(dst, "w", encoding="utf-8")
for r in rows:
    row, kind = X.to_staged_row(r["input"], r["drip"], r["output"]); C[kind] += 1
    if row: fc.write(json.dumps(row, ensure_ascii=False) + chr(10))
fc.close()
usable = sum(v for k, v in C.items() if not k.startswith("stage1"))
print("staged rows:", {k: "%.1f%%" % (100.0 * v / n) for k, v in C.most_common()})
print("usable rows: %d of %d (%.1f%%); dropped because dripper text is not an exact line-subset of the page: %.1f%%" % (usable, n, 100.0 * usable / n, 100.0 * (n - usable) / n))
