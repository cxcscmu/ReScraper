"""Held-out set step 2a (CPU): exclusion list, seeded pick of 2 new held-out pool shards, and candidate pages for them
by the SFT chain (the join the released student's SFT rows went through):
  Dripper step-2 text row k -> step-1 record (ordered, head check within 4, pool_join.head_in) -> drip =
  webkit_txt(main_html) must equal the step-2 text; full = webkit_txt(step-1 input html) (60 s alarm, len >= 20).
Seed filter as the SFT seed sampler: text = webkit_post(step-2 text); drop empty / MARKUP (LID en >= 0.65 is applied in
step 2b, fastText lives in the DCLM data-processing env). Stage-1 feasibility = rescraper_ops.to_staged_row(full, drip,
"<delete>") is not None (the staged builder drops these rows anyway).
Excluded shards: the stage-1 SFT shards of the released student (STAGE1_SFT_STEMS, 1,109 stems), 2 known-bad pool
shards, the shards holding any page of the two-stage refiner's SFT source (step 1b), the 3 existing held-out shards,
and the shards of the other pool SFT/held-out files (step 1 (e)).
usage: s2a_new_shards.py <outdir> <two_stage_stems.txt> <two_stage_src_warc.tsv> <seed>"""
import json, gzip, os, sys, re, signal, hashlib, collections, random, ast, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib")); sys.path.insert(0, os.path.join(HERE, ".."))
from eval_paths import TWO_STAGE_POOL_DIR as T, TEXT_DECISIONS_DIR as DEC, POOL_RAW_DIR as RAW, STAGE1_SFT_STEMS
import rescraper_ops as X, editops as E, pool_join as H, render as B
OUT, T2T_STEMS, T2T_WARC, SEED = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
md5 = lambda s: hashlib.md5(s.encode("utf-8", "surrogatepass")).hexdigest()
def meta(m):
    if isinstance(m, dict): return m
    try: return ast.literal_eval(m)
    except Exception:
        try: return json.loads(m)
        except Exception: return {}
OLD = ["CC-MAIN-20140707234035-00041-ip-10-180-212-248.ec2.internal", "CC-MAIN-20170924171658-20170924191658-00275",
       "CC-MAIN-20200709002952-20200709032952-00415"]
POISON = {"CC-MAIN-20200807231820-20200808021820-00542", "CC-MAIN-20200812053726-20200812083726-00050"}
SFT = set(x.strip() for x in open(STAGE1_SFT_STEMS) if x.strip())
T2T = set(x.strip() for x in open(T2T_STEMS) if x.strip())
OTH = json.load(open(OUT + "/other_sft_shards.json")); OTHER = set(s for v in OTH.values() for s in v if s != "?")
assert len(SFT) == 1109, len(SFT)
print("exclusion: sft", len(SFT), "poison", len(POISON), "two_stage_refiner", len(T2T), "other pool SFT/held-out", len(OTHER), flush=True)
for s in OLD: print("  existing held-out shard", s, "in sft", s in SFT, "poison", s in POISON, "two_stage_refiner", s in T2T, "other-SFT sets:", [k for k, v in OTH.items() if s in v], flush=True)
ls = lambda d, suf: set(f[:-len(suf)] for f in os.listdir(d) if f.endswith(suf))
have = (ls(H.ST1, ".jsonl") & ls(H.ST2, ".jsonl.gz") & ls(RAW, ".jsonl.gz") & ls(T + "/input", ".jsonl.gz") & ls(T + "/best", ".jsonl.gz")
        & ls(T + "/edu", ".jsonl") & ls(T + "/rwrepro_olmo_temp1", ".jsonl") & ls(DEC, ".jsonl"))
have = set(s for s in have if s.startswith("CC-MAIN-"))
excl = SFT | POISON | T2T | set(OLD) | OTHER
cands = sorted(have - excl)
print("pool stems with every artifact", len(have), "-> candidates after exclusion", len(cands), flush=True)
pick = random.Random(SEED).sample(cands, 2)
print("PICK (seed %d, in draw order):" % SEED, pick, flush=True)
json.dump({"seed": SEED, "n_candidates": len(cands), "pick": pick, "excl_counts": {"sft": len(SFT), "poison": len(POISON), "two_stage_refiner": len(T2T),
           "other": len(OTHER), "old": len(OLD)}, "cands_sha1": hashlib.sha1("\n".join(cands).encode()).hexdigest()}, open(OUT + "/pick.json", "w"), indent=1)
open(OUT + "/candidate_stems.txt", "w").write("\n".join(cands) + "\n")
# definitive two-stage-refiner check on the 5 held-out shards through the raw pool metadata (every page, not only
# resiliparse-extracted ones)
want = set(l.split("\t")[2] for l in open(T2T_WARC) if l.split("\t")[2])
chk = {}
for s in OLD + pick:
    n = hit = 0
    with gzip.open(f"{RAW}/{s}.jsonl.gz", "rt", encoding="utf-8") as g:
        for l in g:
            n += 1; hit += meta(json.loads(l)["metadata"]).get("WARC-Record-ID") in want
    chk[s] = {"raw_pages": n, "two_stage_src_warc_hits": hit}; print("  raw WARC check", s, chk[s], flush=True)
json.dump(chk, open(OUT + "/heldout_shards_two_stage_warc_check.json", "w"), indent=1)
# ---------- candidates ----------
B._load_converter()            # import + patch the renderer once, before the worker processes fork
webkit_txt = B.webkit_txt
class TO(Exception): pass
def _init(): signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))
def render_page(h, sec=60):
    signal.alarm(sec)
    try: return webkit_txt(h)
    except Exception: return None
    finally: signal.alarm(0)
def two(a): return render_page(a[0]), render_page(a[1])
from multiprocessing import Pool
pool = Pool(int(os.environ.get("PROCS", "8")), initializer=_init)
fc = open(OUT + "/cand_pre.jsonl", "w", encoding="utf-8"); fh = open(OUT + "/html_new.jsonl", "w", encoding="utf-8")
cid = 0; tot = collections.Counter()
for order, stem in enumerate(pick):
    S = collections.Counter(); t0 = time.time()
    rawmap = {}
    with gzip.open(f"{RAW}/{stem}.jsonl.gz", "rt", encoding="utf-8") as g:
        for j, l in enumerate(g):
            o = json.loads(l); m = meta(o["metadata"]); rawmap.setdefault(md5(o["text"]), (m.get("WARC-Record-ID"), m.get("WARC-Target-URI"), j))
    s1 = []
    with open(f"{H.ST1}/{stem}.jsonl", encoding="utf-8") as f:
        for j, l in enumerate(f):
            r = json.loads(l); mh = r.get("main_html") or ""
            if mh.strip(): s1.append((j, r.get("input") or "", mh, H.norm_head(mh)))
    s2 = [r["text"] for r in H.gz_rows(f"{H.ST2}/{stem}.jsonl.gz")]
    S["step1_nonempty_main_html"] = len(s1); S["step2_rows"] = len(s2)
    pairs = []; i = 0
    for k, t in enumerate(s2):
        jj = i; found = -1
        while jj < len(s1) and jj < i + 4:
            if H.head_in(t, s1[jj][3]): found = jj; break
            jj += 1
        if found < 0: S["no_candidate"] += 1; continue
        i = found + 1; pairs.append((k, t, found))
    rend = pool.map(two, [(s1[f][2], s1[f][1]) for _, _, f in pairs], chunksize=4)
    for (k, t, f), (drip_r, full) in zip(pairs, rend):
        j, html, mh, _ = s1[f]
        if drip_r is None: S["render_fail_main"] += 1; continue
        if drip_r != t: S["text_mismatch"] += 1; continue
        if full is None or len(full) < 20: S["render_fail_or_short"] += 1; continue
        tp = B.webkit_post(t)
        rec = {"cid": cid, "stem": stem, "shard_order": order, "step2_row": k, "step1_idx": j, "in_md5": md5(html),
               "step2_eq_post": tp == t, "empty": not tp.strip(), "markup": bool(B.MARKUP.search(tp)) if tp.strip() else False}
        w = rawmap.get(rec["in_md5"]); rec.update({"warc_id": w[0] if w else None, "url": w[1] if w else None, "raw_row": w[2] if w else None})
        rec["stage1_ok"] = X.to_staged_row(full, tp, "<delete>")[0] is not None if tp.strip() else False
        rec["full"] = full; rec["drip"] = tp
        fc.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.write(json.dumps({"cid": cid, "html": html}, ensure_ascii=False) + "\n")
        S["ok"] += 1; S["stage1_ok"] += rec["stage1_ok"]; S["markup"] += rec["markup"]; S["empty"] += rec["empty"]; S["no_warc"] += w is None
        cid += 1
    print("stem", stem, dict(S), "%.0fs" % (time.time() - t0), flush=True); tot.update(S)
fc.close(); fh.close(); pool.close()
print("TOTAL", dict(tot), flush=True)
print("STEP2A_OK")
