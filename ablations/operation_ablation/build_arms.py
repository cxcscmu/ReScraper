"""Operation ablation: build the counterfactual arms from the raw sidecar and verify the full arm.

usage: build_arms.py <ABL dir> [procs] [--smoke] [--compare-release <release text dir>]
reads  <ABL>/raw/<stem>_raw.jsonl.gz                 (infer_pool_with_raw.py, one row per step1 record)
       <ABL>/full/text/<stem>_processed.jsonl.gz     (same job, written by the UNMODIFIED release writer lines)
writes <ABL>/{nodel,noedit,norw}/text/<stem>_processed.jsonl.gz   (atomic, skipped if present = resumable)
       <ABL>/build_stats.json

Arm rule. A page = raw row with a generation whose executor tag parsed (tag not None), in step1 order.
  full    = the released writer: drop tag <delete> or empty final, else {"text": final, "e2e_tag": tag}
  nodel   pages tagged <delete>  -> {"text": extracted, "e2e_tag": "<extract>", "ablation_from": "<delete>"}
  noedit  pages tagged <refine>  -> same, with the <edit> removals not applied
  norw    pages tagged <rewrite> -> same, extracted text instead of the paraphrase
  every other page is written exactly as in full; a counterfactual page whose extracted text is empty is dropped
  (the writer's own empty-text rule). e2e_tag <extract> keeps the page past postfilter rule A3; the
  postfilter writes only text + e2e_tag, so ablation_from never reaches text_clean/dedup/tokenize.
Checks (any failure -> exit 3, message BUILD_FAIL):
  F1  full rows rebuilt from raw == decompressed full/text shard, byte for byte
  F2  raw in_full flags == rows actually written
  F3  every <extract> page: final == extracted (the executor's keep path is apply_ops(ninp, ops1))
      drop build: an F3 page (40 in the pool) is a malformed generation whose final is
      program residue ("rm 1-28 ... <extract>") or empty. It is DROPPED from every arm, including full: the full/text
      shard of an affected stem is rewritten without it and that stem's three arm shards are rebuilt. F1 then
      accepts full/text on disk == the release writer rows (first run) or == those rows minus the F3 pages (re-run).
  F4  every arm, read back from disk: rows without ablation_from == full rows minus the arm's target-tag pages (in
      order), rows with ablation_from == the non-empty extracted texts of the target-tag pages (in order)
  --smoke (raw rows carry ninp):
  S1  the UNMODIFIED release loop re-run on (ninp, gen) with e2e_ops.body_from_prediction_dfirst writes exactly
      the full/text bytes, and reproduces every raw (tag, final)
  S2  extracted == what the UNMODIFIED executor returns for the same generation with its decision line set to <keep>
  SMOKE gate: class mix within broad bounds of the release, <=2% malformed, full-arm rows within 15% of the
      released corpus on the same shards (the array is submitted afterok:smoke)
"""
import collections, gzip, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import e2e_ops as X, editops as E

ARMS = {"nodel": "<delete>", "noedit": "<refine>", "norw": "<rewrite>"}
STD = ("<extract>", "<refine>", "<rewrite>", "<delete>")


def row(text, tag, src=None):
    d = {"text": text, "e2e_tag": tag}
    if src is not None:
        d["ablation_from"] = src
    return json.dumps(d, ensure_ascii=False) + "\n"


def read_lines(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return f.readlines()


def write_atomic(p, lines):
    tmp = "%s.tmp%d" % (p, os.getpid())
    with gzip.open(tmp, "wt", encoding="utf-8") as g:
        g.writelines(lines)
    os.replace(tmp, p)


def one(args):
    abl, stem, smoke, release = args
    S = collections.Counter(); fails = []
    try:
        recs = [json.loads(l) for l in read_lines(f"{abl}/raw/{stem}_raw.jsonl.gz")]
        full_disk = read_lines(f"{abl}/full/text/{stem}_processed.jsonl.gz")
    except Exception as e:
        return S, [f"{stem}: unreadable {type(e).__name__}: {e}"], None
    S["shards"] += 1; S["raw_rows"] += len(recs)
    if [r["idx"] for r in recs] != sorted(r["idx"] for r in recs) or len({r["idx"] for r in recs}) != len(recs):
        fails.append(f"{stem}: raw idx not strictly increasing")
    full, arms, flags = [], {a: [] for a in ARMS}, []
    full_writer, raw_flags = [], []  # v2: rows/flags exactly as the release writer produced them
    expect_keep = {a: [] for a in ARMS}; expect_cf = {a: [] for a in ARMS}
    for r in recs:
        if "skip" in r:
            S["skip_" + r["skip"]] += 1; continue
        S["generated"] += 1
        S["finish_" + str(r.get("finish"))] += 1
        tag = r.get("tag")
        if tag is None:
            S["parse_fail"] += 1; flags.append(False); raw_flags.append(False); continue
        S["tag" + (tag if tag in STD else "_other")] += 1
        final, ext = r["final"], r.get("extracted")
        if ext is None:
            fails.append(f"{stem}:{r['idx']} extracted missing"); ext = ""
        in_full = not (tag == "<delete>" or not (final or "").strip())
        raw_flags.append(in_full)
        if in_full:
            full_writer.append(row(final, tag))
        if tag == "<extract>" and final != ext:
            S["f3_pages"] += 1; S["f3_dropped_rows"] += int(in_full)
            print(f"F3_DROP {stem}:{r['idx']} final_chars={len(final or '')} extracted_chars={len(ext)} in_full={in_full}", flush=True)
            flags.append(False); continue
        flags.append(in_full)
        if in_full:
            full.append(row(final, tag))
        for a, t in ARMS.items():
            if tag == t:
                if ext.strip():
                    arms[a].append(row(ext, "<extract>", t)); expect_cf[a].append(ext)
                    S[a + "_cf_rows"] += 1; S[a + "_cf_chars"] += len(ext)
                else:
                    S[a + "_cf_empty_dropped"] += 1
            elif in_full:
                arms[a].append(row(final, tag)); expect_keep[a].append(full[-1])
        if smoke:
            ninp = r["ninp"]
            t2, x2 = X.body_from_prediction_dfirst(ninp, r["gen"])
            if (t2, x2) != (tag, final):
                fails.append(f"{stem}:{r['idx']} S1 executor re-run differs")
            # S2: the UNMODIFIED executor, asked what this page would be had the decision been <keep>
            lines = r["gen"].strip().split("\n"); d = lines[0].strip() if lines else ""
            if d in X.DFIRST_TAGS:
                t3, x3 = X.body_from_prediction_dfirst(ninp, "<keep>\n" + "\n".join(lines[1:]))
                ok = t3 == "<extract>" and x3 == ext
            else:
                S["S2_staged_fallback_pages"] += 1
                ok = E.apply_ops(ninp, X.parse_staged(r["gen"])[0]) == ext
            if ok: S["S2_ok"] += 1
            else: fails.append(f"{stem}:{r['idx']} S2 extracted differs from the executor's keep reading")
    S["full_rows"] += len(full); S["full_chars"] += sum(len(json.loads(l)["text"]) for l in full)
    rewrite_arms = False
    if full_disk == full_writer and full != full_writer:
        write_atomic(f"{abl}/full/text/{stem}_processed.jsonl.gz", full); S["f3_full_shards_rewritten"] += 1
        rewrite_arms = True
        print(f"F3_REWRITE {stem}: full/text {len(full_writer)} -> {len(full)} rows", flush=True)
    elif full_disk != full:
        fails.append(f"{stem}: F1 full rebuilt ({len(full)} rows) != full/text on disk ({len(full_disk)} rows)")
    if [r.get("in_full") for r in recs if "skip" not in r] != raw_flags:
        fails.append(f"{stem}: F2 in_full flags disagree")
    if smoke:
        # S1 again, literally the release writer loop over (ninp, gen) in order, compared as bytes
        sim = []
        for r in recs:
            if "skip" in r: continue
            try: tag, text = X.body_from_prediction_dfirst(r["ninp"], r["gen"])
            except Exception: continue
            if tag == "<delete>" or not (text or "").strip(): continue
            sim.append(json.dumps({"text": text, "e2e_tag": tag}, ensure_ascii=False) + "\n")
        if "".join(sim).encode("utf-8") != "".join(full_disk).encode("utf-8"):
            fails.append(f"{stem}: S1 release writer loop bytes != full/text")
        else:
            S["S1_bytes_identical_shards"] += 1
    for a in ARMS:
        dst = f"{abl}/{a}/text/{stem}_processed.jsonl.gz"
        if rewrite_arms or not os.path.exists(dst):
            write_atomic(dst, arms[a]); S[a + "_written"] += 1
        else:
            S[a + "_existing"] += 1
        back = read_lines(dst)
        S[a + "_rows"] += len(back)
        keep = [l for l in back if "ablation_from" not in json.loads(l)]
        cf = [json.loads(l) for l in back if "ablation_from" in json.loads(l)]
        if keep != expect_keep[a] or [c["text"] for c in cf] != expect_cf[a] or any(c["ablation_from"] != ARMS[a] or c["e2e_tag"] != "<extract>" for c in cf):
            fails.append(f"{stem}: F4 arm {a} on disk differs from the rule")
        if [l for l in full if json.loads(l)["e2e_tag"] != ARMS[a]] != expect_keep[a]:
            fails.append(f"{stem}: F4 arm {a} kept rows != full minus {ARMS[a]}")
    cmp = None
    if release:
        p = f"{release}/{stem}_processed.jsonl.gz"
        if os.path.exists(p):
            rel = [json.loads(l) for l in read_lines(p)]
            cmp = {"stem": stem, "release_rows": len(rel), "ours_rows": len(full),
                   "release_tags": dict(collections.Counter(r.get("e2e_tag") for r in rel)),
                   "ours_tags": dict(collections.Counter(json.loads(l)["e2e_tag"] for l in full)),
                   "release_chars": sum(len(r["text"]) for r in rel), "ours_chars": S["full_chars"]}
    return S, fails, cmp


def main():
    abl = sys.argv[1]
    procs = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 16
    smoke = "--smoke" in sys.argv
    assert not smoke, "the drop build is the post-array build; it does not implement the pre-array --smoke gate"
    release = sys.argv[sys.argv.index("--compare-release") + 1] if "--compare-release" in sys.argv else None
    for a in ARMS:
        os.makedirs(f"{abl}/{a}/text", exist_ok=True)
    stems = sorted(f[:-len("_raw.jsonl.gz")] for f in os.listdir(f"{abl}/raw") if f.endswith("_raw.jsonl.gz"))
    have_full = {f[:-len("_processed.jsonl.gz")] for f in os.listdir(f"{abl}/full/text") if f.endswith("_processed.jsonl.gz")}
    print(f"ABL={abl} raw_shards={len(stems)} full_shards={len(have_full)} procs={procs} smoke={smoke}", flush=True)
    T = collections.Counter(); fails = []; cmps = []
    if set(stems) != have_full:
        fails.append(f"raw/full shard sets differ: raw-only={sorted(set(stems)-have_full)[:5]} full-only={sorted(have_full-set(stems))[:5]}")
    stems = [s for s in stems if s in have_full]
    from multiprocessing import Pool
    with Pool(procs) as p:
        for n, (S, f, c) in enumerate(p.imap_unordered(one, [(abl, s, smoke, release) for s in stems], chunksize=4), 1):
            T.update(S); fails.extend(f)
            if c: cmps.append(c)
            if n % 1000 == 0:
                print(f"  {n}/{len(stems)} fails={len(fails)}", flush=True)
    gen_ok = sum(T["tag" + t] for t in STD) + T["tag_other"]
    mix = {t: round(T["tag" + t] / max(gen_ok, 1), 4) for t in STD}
    mix["other"] = round(T["tag_other"] / max(gen_ok, 1), 4)
    out = {"counters": dict(T), "mix_of_parsed_pages": mix, "n_fail": len(fails), "fails_head": fails[:30]}
    if cmps:
        agg = {"shards": len(cmps), "release_rows": sum(c["release_rows"] for c in cmps), "ours_rows": sum(c["ours_rows"] for c in cmps),
               "release_chars": sum(c["release_chars"] for c in cmps), "ours_chars": sum(c["ours_chars"] for c in cmps),
               "release_tags": dict(sum((collections.Counter(c["release_tags"]) for c in cmps), collections.Counter())),
               "ours_tags": dict(sum((collections.Counter(c["ours_tags"]) for c in cmps), collections.Counter()))}
        out["release_compare_total"] = agg
        print("RELEASE_COMPARE " + json.dumps(agg, ensure_ascii=False), flush=True)
    print("BUILD_STATS " + json.dumps(out, ensure_ascii=False), flush=True)
    if cmps:
        out["release_compare"] = sorted(cmps, key=lambda c: c["stem"])
    json.dump(out, open(f"{abl}/build_stats.json", "w"), indent=1)
    for a, t in ARMS.items():
        print(f"ARM {a}: rows={T[a + '_rows']} (full {T['full_rows']}) counterfactual_rows={T[a + '_cf_rows']} "
              f"cf_chars={T[a + '_cf_chars']} empty_dropped={T[a + '_cf_empty_dropped']}", flush=True)
    print(f"MIX {mix}", flush=True)
    print(f"F3 pages={T['f3_pages']} dropped_rows={T['f3_dropped_rows']} full_shards_rewritten={T['f3_full_shards_rewritten']}", flush=True)
    if smoke:
        # Smoke gate (the inference array hangs off afterok:smoke). Broad bounds around the released run's mix
        # (extract ~29%, refine ~21%, rewrite ~9%, delete ~39% of routed pages) - 8 shards at T=1 are noisy.
        B = {"<extract>": (0.15, 0.45), "<refine>": (0.08, 0.35), "<rewrite>": (0.03, 0.20), "<delete>": (0.25, 0.55)}
        for t, (lo, hi) in B.items():
            if not lo <= mix[t] <= hi:
                fails.append(f"SMOKE mix {t}={mix[t]} outside [{lo},{hi}]")
        if (T["tag_other"] + T["parse_fail"]) > 0.02 * max(T["generated"], 1):
            fails.append(f"SMOKE other+parse_fail={T['tag_other'] + T['parse_fail']} > 2% of {T['generated']}")
        if T["S1_bytes_identical_shards"] != T["shards"] or T["S2_ok"] != T["generated"] - T["parse_fail"]:
            fails.append(f"SMOKE S1 {T['S1_bytes_identical_shards']}/{T['shards']} shards, S2 {T['S2_ok']}/{T['generated'] - T['parse_fail']} pages")
        if cmps:
            rr = sum(c["release_rows"] for c in cmps); oo = sum(c["ours_rows"] for c in cmps)
            print(f"SMOKE full-arm rows {oo} vs released corpus {rr} on the same {len(cmps)} shards ({oo / max(rr, 1):.3f}x)", flush=True)
            if not 0.85 <= oo / max(rr, 1) <= 1.15:
                fails.append(f"SMOKE full-arm rows {oo} vs release {rr}: ratio outside [0.85, 1.15]")
        elif release:
            fails.append("SMOKE no release shards to compare")
    if fails:
        for f in fails[:30]:
            print("FAIL " + f, flush=True)
        print(f"BUILD_FAIL n={len(fails)}", flush=True); sys.exit(3)
    print(f"BUILD_OK shards={T['shards']}", flush=True)


if __name__ == "__main__":
    main()
