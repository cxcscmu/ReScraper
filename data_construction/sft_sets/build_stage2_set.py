"""Stage-2 SFT set: enough <rewrite> for the student to learn the tag (the paper uses scheme edu1, rw_frac 0.30,
rewrite_dir = two_stage_pool/rwrepro_olmo_temp1; result 131,484 rows: 39,445 rewrite / 35,189 keep /
32,239 delete / 24,611 edit).

The 1.38M set is 0.11% <rewrite> (1,587 rows) and the resulting student emits the tag exactly zero
times over 161,260 pages of pool inference - it learned to never rewrite. So the class is raised to
RW_FRAC by subsampling everything else rather than duplicating the rewrite rows, which is why the set
comes out smaller.

Two label schemes, crossed with two rewrite sources, give the 2x2:
  scheme rescue : edu >= 1.5 -> <keep> verbatim, 1.0 <= edu < 1.5 -> <rewrite>
  scheme edu1   : edu >= 1   -> <rewrite>            (no keep branch, no DataMan, no output gate)
  scheme allrw  : edu >= 0.5 -> <rewrite>            (the rewrite-production gate of rewrite_pool.py: int_edu>=1 == round(edu)>=1)
  scheme rw15   : edu >= 1.5 -> <rewrite>            (only the band rescue keeps verbatim; 1.0-1.5 stays deleted)
  source        : two_stage_pool/rwrepro_olmo_temp1 (RePro upstream sampling, T=1.0; used in the paper)

Non-rewrite rows are subsampled stratified by their own leading tag, so keep/edit/delete keep the
proportions they had in the parent set and only the rewrite share moves.
usage: build_stage2_set.py <sft_in> <join_glob[,..]> <decisions_dir> <rewrite_dir> <scheme> <rw_frac> <out.jsonl>
"""
import json, sys, glob, os, gzip, random, collections

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import editops as E

LEAD = ("<keep>", "<edit>", "<delete>", "<rewrite>")


def main():
    src, joinpat, decdir, rwdir, scheme, frac, dst = sys.argv[1:8]
    frac = float(frac)
    assert scheme in ("rescue", "edu1", "allrw", "rw15"), scheme
    MINEDU = {"rescue": 1.0, "edu1": 1.0, "allrw": 0.5, "rw15": 1.5}[scheme]

    edu = {}
    for f in sorted(glob.glob(os.path.join(decdir, "*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            edu[(r["stem"], int(r["row"]))] = float(r["edu"])
    print("edu scores for deleted pages: %d" % len(edu), flush=True)

    # rewrite text, keyed the same way; only the shards this SFT set can draw from matter
    need = {s for s, _ in edu}
    rw = {}
    for stem in need:
        p = os.path.join(rwdir, stem + ".jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p, encoding="utf-8"):
            o = json.loads(l)
            t = (o.get("text") or "").strip()
            if t:
                rw[(stem, int(o["i"]))] = t
    print("rewrite texts available: %d (from %s)" % (len(rw), rwdir), flush=True)

    joins = []
    for pat in joinpat.split(","):
        pat = pat.strip()
        if pat:
            joins += sorted(glob.glob(pat))
    assert joins, ("no join files matched", joinpat)
    key = {}
    for i, f in enumerate(joins):
        for l in open(f, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            k = (r.get("stem"), int(r.get("row", -1)))
            if k in edu:
                key[E.number_lines(r["input"])] = k
        print("  %3d/%d matched so far %d" % (i + 1, len(joins), len(key)), flush=True)

    rows, S = [], collections.Counter()
    with open(src, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            t = o["output"]
            if t.startswith("<delete>"):
                k = key.get(o["input"])
                if k:
                    e = edu[k]
                    parts = t.split("<delete>", 2)
                    ops = parts[1] if len(parts) > 2 else "\n<extract>\n"
                    if scheme == "rescue" and e >= 1.5:
                        o["output"] = "<keep>" + ops + "<keep>"; S["flip_keep"] += 1
                    elif k in rw and e >= MINEDU:
                        # scheme rescue already took e >= 1.5 above, so this is its 1.0-1.5 band;
                        # scheme edu1 has no keep branch and takes everything from 1.0 up.
                        o["output"] = "<rewrite>" + ops + "<rewrite>\n" + rw[k]; S["flip_rewrite"] += 1
                    else:
                        S["unchanged"] += 1
                else:
                    S["no_decision"] += 1
            rows.append(o)

    lead = lambda o: next((x for x in LEAD if o["output"].startswith(x)), "(other)")
    rwrows = [o for o in rows if lead(o) == "<rewrite>"]
    other = collections.defaultdict(list)
    for o in rows:
        if lead(o) != "<rewrite>":
            other[lead(o)].append(o)
    n_other = int(round(len(rwrows) * (1 - frac) / frac))
    tot_other = sum(len(v) for v in other.values())
    random.seed(17)
    picked = []
    for tag, v in other.items():
        take = min(len(v), int(round(n_other * len(v) / tot_other)))
        picked += random.sample(v, take)
    out = rwrows + picked
    random.shuffle(out)
    with open(dst + ".tmp", "w", encoding="utf-8") as g:
        for o in out:
            g.write(json.dumps(o, ensure_ascii=False) + "\n")
    os.replace(dst + ".tmp", dst)
    c = collections.Counter(lead(o) for o in out)
    print("relabel: %s" % dict(S))
    print("wrote %d rows -> %s" % (len(out), dst))
    for t, n in c.most_common():
        print("  %-10s %8d %6.2f%%" % (t, n, 100.0 * n / len(out)))
    assert c["<rewrite>"] > 0, "no rewrite rows - check the rewrite dir"
    print("BUILD_STAGE2_SET_OK")


if __name__ == "__main__":
    main()
