"""Row-by-row comparison of two Stage-1 SFT files (same row order expected): first tag of each,
whether the inputs are identical, and the tag transition matrix a -> b. Used to check that the
Stage-1 set of Table 6 differs from its no-rewrite base set only by the rescue relabelling
(<delete> -> <keep> / <rewrite> on the same input).
CPU only; run it where the SFT files live.
usage: python compare_sft_tags.py <a.jsonl> <b.jsonl> <out.json>
"""
import collections, hashlib, json, sys, time

pa, pb, outp = sys.argv[1:4]
trans = collections.Counter(); same_input = 0; n = 0; t0 = time.time()
ex = collections.defaultdict(list)
with open(pa, "rb") as fa, open(pb, "rb") as fb:
    for la, lb in zip(fa, fb):
        a = json.loads(la); b = json.loads(lb); n += 1
        ta = (a["output"] or "").strip().split("\n")[0].strip()
        tb = (b["output"] or "").strip().split("\n")[0].strip()
        si = a["input"] == b["input"]; same_input += si
        trans[ta + " -> " + tb + (" (same input)" if si else " (DIFFERENT input)")] += 1
        if ta != tb and len(ex[ta + tb]) < 3:
            ex[ta + tb].append({"row": n - 1, "a_out_head": a["output"][:300], "b_out_head": b["output"][:300]})
    rest_a = sum(1 for _ in fa); rest_b = sum(1 for _ in fb)
res = {"a": pa, "b": pb, "rows_compared": n, "extra_rows_a": rest_a, "extra_rows_b": rest_b,
       "same_input": same_input, "transitions": dict(trans), "examples": ex, "seconds": round(time.time() - t0)}
json.dump(res, open(outp, "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
print("PAIR_DONE")
