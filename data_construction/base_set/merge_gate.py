#!/usr/bin/env python3
"""Gates + union for the no-rewrite base set.

usage:
  merge_gate.py union <dst> <src1> <src2> ...   concatenate + seeded shuffle, then check
  merge_gate.py check <file>                    structural gate only

check = every row's output must be <DEC>\n<extract>\n...<DEC>[payload] with DEC in
keep|edit|delete (norw: rewrite must be 0), the stage-1 block present, and the leading
decision equal to the trailing one. Prints the tag mix and three truncated sample rows.
"""
import json, sys, random, collections, os, re

DEC = ("keep", "edit", "delete", "rewrite")
LEAD = re.compile(r"^<(keep|edit|delete|rewrite)>\n<extract>\n")


def check(path):
    n = bad_lead = bad_ext = mismatch = 0
    mix = collections.Counter()
    samples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            out = d["output"]
            n += 1
            m = LEAD.match(out)
            if not m:
                bad_lead += 1
                if bad_lead <= 3:
                    print("BAD_LEAD row", i, repr(out[:120]))
                continue
            lead = m.group(1)
            mix[lead] += 1
            rest = out[m.end():]
            # trailing decision: the last line that is a bare <DEC> tag
            tail = [l for l in rest.split("\n") if l.strip() in ["<%s>" % t for t in DEC]]
            if not tail:
                bad_ext += 1
                if bad_ext <= 3:
                    print("NO_STAGE2 row", i, repr(out[-120:]))
            elif tail[-1].strip() != "<%s>" % lead:
                mismatch += 1
                if mismatch <= 3:
                    print("LEAD!=TAIL row", i, lead, tail[-1])
            if len(samples) < 3 and i % 7919 == 0:
                samples.append((i, d["input"][:200], out[:400]))
    print("rows", n, "tag mix", dict(mix),
          {k: "%.1f%%" % (100.0 * v / max(1, n)) for k, v in mix.most_common()})
    print("bad_lead", bad_lead, "no_stage2", bad_ext, "lead!=tail", mismatch)
    for i, inp, out in samples:
        print("--- sample row", i, "input[:200] ---");  print(inp)
        print("--- output[:400] ---");                  print(out)
    ok = (bad_lead == 0 and bad_ext == 0 and mismatch == 0 and mix["rewrite"] == 0 and n > 0)
    print("GATES_OK" if ok else "GATES_FAIL")
    return 0 if ok else 1


def union(dst, srcs):
    lines = []
    for s in srcs:
        c = 0
        with open(s, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(line)
                    c += 1
        print("read", c, "rows from", os.path.basename(s), flush=True)
    random.Random(0).shuffle(lines)
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as g:
        g.writelines(lines)
    os.replace(tmp, dst)
    print("union rows", len(lines), "->", dst, flush=True)
    return check(dst)


if __name__ == "__main__":
    if sys.argv[1] == "union":
        sys.exit(union(sys.argv[2], sys.argv[3:]))
    sys.exit(check(sys.argv[2]))
