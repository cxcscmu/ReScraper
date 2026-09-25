#!/usr/bin/env python3
"""Gate for the decision-first set: round-trip sampled rows through the executor and assert
dfirst(new target) == staged(old target), plus an rwstrict scan of the rewrite bodies.
usage: verify_dfirst.py <staged.jsonl> <dfirst.jsonl> [n_sample=4000]"""
import json, os, sys, random, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import e2e_ops as X, rwstrict

src, dst = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

MAP = {"<keep>": "<extract>", "<edit>": "<refine>", "<delete>": "<delete>", "<rewrite>": "<rewrite>"}
bad_rt = bad_line1 = dirty = 0
hist = collections.Counter()
show = []
rng = random.Random(0)
with open(src) as f1, open(dst) as f2:
    for i, (a, b) in enumerate(zip(f1, f2)):
        ra, rb = json.loads(a), json.loads(b)
        if ra["input"] != rb["input"]:
            print("FATAL: input drift at row", i); sys.exit(1)
        old, new = ra["output"], rb["output"]
        l1 = new.split("\n", 1)[0]
        if l1 not in MAP:
            bad_line1 += 1
        hist[l1] += 1
        # rwstrict applies to rephraser output only: the text after <rewrite>
        if "<rewrite>" in new:
            body = new.split("<rewrite>", 1)[1].strip()
            if body and rwstrict.is_dirty(body):
                dirty += 1
        if i % max(1, 87532 // N) == 0:
            ninp = rb["input"]
            t_old, x_old = X.body_from_prediction_staged(ninp, old)
            t_new, x_new = X.body_from_prediction_dfirst(ninp, new)
            if (t_old, x_old) != (t_new, x_new):
                bad_rt += 1
                if len(show) < 3:
                    show.append((i, t_old, t_new, x_old[:120], x_new[:120]))
            if MAP[l1] != t_old and l1 in MAP:
                bad_rt += 1
print("line-1 tag mix        ", dict(hist))
print("rows whose line 1 is not a bare decision tag:", bad_line1)
print("round-trip mismatches (dfirst vs staged), sampled:", bad_rt)
print("rewrite bodies flagged dirty by rwstrict:", dirty)
for s in show: print("  MISMATCH", s)
print("VERDICT", "PASS" if (bad_line1 == 0 and bad_rt == 0 and dirty == 0) else "FAIL")
