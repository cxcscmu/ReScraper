#!/usr/bin/env python3
"""Decision-first targets.

Staged target shape:
    <extract>\n{rm a-b lines}\n<DEC>[\n{payload}]      DEC in keep|edit|delete|rewrite

The decision tag is emitted AFTER the stage-1 op list, so the model predicts it from a
position where it has already committed to an extraction. This rewrites every target as

    <DEC>\n<extract>\n{rm lines}\n<DEC>[\n{payload}]

The decision now also appears at position 0, predicted from the input alone. The rest of
the target is byte-identical, so the staged reader keeps finding the tag in its current
slot and no payload text is created, altered or reordered.
usage: build_dfirst.py <staged.jsonl> <dfirst.jsonl>
"""
import json, sys, re, collections

DEC = ("<keep>", "<edit>", "<delete>", "<rewrite>")
DEC_RE = re.compile(r"^(<keep>|<edit>|<delete>|<rewrite>)$", re.M)

def main(src, dst):
    hist = collections.Counter()
    n = nodec = 0
    changed_payload = 0
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            d = json.loads(line)
            out = d["output"]
            m = DEC_RE.search(out)
            if not m:
                nodec += 1
                hist["<NONE>"] += 1
                g.write(json.dumps(d) + "\n")
                n += 1
                continue
            dec = m.group(1)
            hist[dec] += 1
            new = dec + "\n" + out
            # invariant: the original target must survive verbatim as a suffix
            if not new.endswith(out) or len(new) != len(dec) + 1 + len(out):
                changed_payload += 1
            d["output"] = new
            g.write(json.dumps(d) + "\n")
            n += 1
    print("rows", n)
    print("tag mix", dict(hist))
    print("rows with no decision tag", nodec)
    print("rows whose original target was not preserved verbatim", changed_payload)
    print("VERDICT", "OK" if changed_payload == 0 else "FAIL")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
