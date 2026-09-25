#!/usr/bin/env python3
"""Provenance gate: the shards (`stem`) a training set was drawn from must be disjoint from the held-out
shards. Content overlap is expected (web pages recur across shards) and is NOT a leak.

The stems live in wanted.jsonl (one record per stem, {"stem": ..., "rows": [...]}), which is what selection
used; the held-out side may be a wanted.jsonl or a plain list of stems (one per line).
usage: check_stem_disjoint.py <train wanted.jsonl> <held-out wanted.jsonl | stems.txt>
"""
import json, sys, collections

def load(path):
    st = collections.Counter()
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        if not l.lstrip().startswith("{"):
            st[l.strip()] += 1; continue
        d = json.loads(l)
        s = d.get("stem", "?")
        st[s] += len(d.get("rows", [])) or 1
    return st

tr = load(sys.argv[1]); gd = load(sys.argv[2])
inter = sorted(set(tr) & set(gd))
print("train stems %d (%d rows) | held-out stems %d (%d rows)" % (len(tr), sum(tr.values()), len(gd), sum(gd.values())))
print("held-out stems:", sorted(gd))
print("intersection:", len(inter), inter[:10])
print("PROV_DISJOINT_OK" if not inter else "PROV_OVERLAP_FAIL")
sys.exit(0 if not inter else 1)
