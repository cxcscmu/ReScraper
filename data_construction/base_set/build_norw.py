"""No-rewrite base targets: every <rewrite> row of a decision-first file becomes a bare <delete> (the refining
teacher's decision before any rescue); stage-1 <extract> ops and every other row stay byte-identical. The result
teaches "Dripper + refining teacher" only; <rewrite> targets are added by data_construction/sft_sets/.
usage: build_norw.py <src> <dst>
"""
import json, sys, collections, re
src, dst = sys.argv[1], sys.argv[2]
before, after = collections.Counter(), collections.Counter()
n = conv = 0
TAG = re.compile(r"^<(keep|edit|delete|rewrite)>\n")
with open(src, encoding="utf-8") as f, open(dst + ".tmp", "w", encoding="utf-8") as g:
    for line in f:
        d = json.loads(line); out = d["output"]; n += 1
        m = TAG.match(out); before[m.group(1) if m else "NONE"] += 1
        if m and m.group(1) == "rewrite":
            parts = out.split("<rewrite>", 2)          # ['', '\n<extract>\n<ops>\n', '\n<paraphrase>']
            assert len(parts) == 3 and parts[0] == "" and parts[1].startswith("\n<extract>"), repr(out[:120])
            out = "<delete>" + parts[1] + "<delete>"
            conv += 1
        m_after = TAG.match(out); after[m_after.group(1) if m_after else "NONE"] += 1
        d["output"] = out
        g.write(json.dumps(d, ensure_ascii=False) + "\n")
import os; os.replace(dst + ".tmp", dst)
print("rows", n, "converted", conv); print("before", dict(before)); print("after", dict(after))
assert after["rewrite"] == 0 and after["NONE"] == 0 and sum(after.values()) == n
print("NORW_OK")
