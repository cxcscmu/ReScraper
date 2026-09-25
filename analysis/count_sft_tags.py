"""Count the operation tags of an SFT set (Table 6, tab:sft-composition, Stage 1 columns).

Streams the Stage-1 SFT set written by data_construction/sft_sets/build_stage1_set.py
(rows {input, output}; output line 1 = decision tag,
then the <extract> block, then the repeated decision + payload) and counts
  first     : line 1 of output (the decision the model is trained to emit first)
  tail      : the tag after the <extract> block (should repeat line 1)
  first_x_tail : agreement of the two
  rewrite payload stats (rows whose first tag is <rewrite>: payload chars)
The Stage-1 "Targets" column is `first_tag`. CPU only; run it where the SFT file lives.
usage: python count_sft_tags.py <jsonl> <out.json>
"""
import collections, json, sys, time

path, outp = sys.argv[1], sys.argv[2]
first = collections.Counter(); tail = collections.Counter(); both = collections.Counter()
bad = 0; n = 0; rw_chars = []; t0 = time.time()
with open(path, "rb") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            bad += 1
            continue
        n += 1
        out = d.get("output") or ""
        lines = out.strip().split("\n")
        t1 = lines[0].strip() if lines else ""
        first[t1 if t1.startswith("<") and len(t1) < 20 else "<non-tag>"] += 1
        # tag after the <extract> block (lines 2..): skip "rm"/"sub"/blank lines
        t2 = "<none>"
        if len(lines) > 1 and lines[1].strip() == "<extract>":
            i = 2
            while i < len(lines):
                s = lines[i].strip()
                if not s or s.startswith("rm ") or s.startswith("sub "):
                    i += 1
                    continue
                t2 = s if (s.startswith("<") and len(s) < 20) else "<non-tag>"
                break
        tail[t2] += 1
        both[t1 + " | " + t2] += 1
        if t1 == "<rewrite>":
            rw_chars.append(len(out))
        if n % 200000 == 0:
            print(n, round(time.time() - t0), dict(first), flush=True)
rw_chars.sort()
res = {"file": path, "rows": n, "unparseable_json": bad, "first_tag": dict(first), "tail_tag": dict(tail),
       "first_and_tail": dict(both),
       "rewrite_rows_output_chars": {"n": len(rw_chars), "p50": rw_chars[len(rw_chars) // 2] if rw_chars else None},
       "seconds": round(time.time() - t0)}
json.dump(res, open(outp, "w"), indent=1)
print(json.dumps(res, indent=1))
print("COUNT_DONE")
