"""SFT seed pages sampled from Dripper's extracted text of the pool (step2_text), ~1.6M pages.

Shards: every pool shard except the held-out shards (HELDOUT_SHARDS) and any further excluded shards
(EXCLUDE_SHARDS); filters: webkit_post cleanup, non-empty, no residual markup (render.MARKUP), fastText
lid.176 English >= 0.65; dedup by a hash of the first 1,500 normalized characters, also against the inputs of
any earlier SFT files listed in DEDUP_AGAINST. Shard order and page order are fixed seeds (11, 12).
Writes $SFT_DIR/seed/seed_pages.jsonl {"i","stem","row","output": dripper text} in shuffled order.

In our run the excluded shards were the 3 held-out shards plus the ~100 shards of an earlier, smaller SFT pool,
and DEDUP_AGAINST listed the four earlier SFT sets of preliminary experiments.
env:   WORK_DIR, SFT_DIR (default $WORK_DIR/sft_data), HELDOUT_SHARDS / EXCLUDE_SHARDS (text files, one shard
       stem per line), DEDUP_AGAINST (comma-separated jsonl files with an "input" field; optional)
usage: sample_seed_pages.py <target_n> [workers]"""
import json, gzip, os, sys, random, hashlib, collections
from multiprocessing import get_context
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import render as B
from pool_join import ST2
R = os.environ["WORK_DIR"]; D = os.environ.get("SFT_DIR", R + "/sft_data"); W = D + "/seed"
target = int(sys.argv[1]); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 48
norm = lambda s: hashlib.sha1(" ".join(s.split())[:1500].lower().encode()).hexdigest()
def seen_hashes():
    s = set()
    for p in [x for x in os.environ.get("DEDUP_AGAINST", "").split(",") if x.strip()]:
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"): s.add(norm(json.loads(l)["input"]))
    return s
_lid = None
def one(stem):
    global _lid
    import fasttext
    if _lid is None: _lid = fasttext.load_model(B.LID_PATH)
    out = []; c = collections.Counter()
    try:
        with gzip.open(f"{ST2}/{stem}.jsonl.gz", "rt", encoding="utf-8") as f:
            for row, l in enumerate(f):
                t = B.webkit_post(json.loads(l).get("text") or "")
                if not t.strip(): c["empty"] += 1; continue
                if B.MARKUP.search(t): c["markup"] += 1; continue
                lab, prob = _lid.predict(t.replace("\n", " ")[:2000], k=1)
                if lab[0] != "__label__en" or prob[0] < 0.65: c["non_en"] += 1; continue
                c["kept"] += 1; out.append((stem, row, t))
    except Exception as e:
        c["error"] += 1
    return out, c
if __name__ == "__main__":
    os.makedirs(W, exist_ok=True)
    read_stems = lambda p: set(l.strip() for l in open(p, encoding="utf-8") if l.strip()) if p and os.path.exists(p) else set()
    held = read_stems(os.environ.get("HELDOUT_SHARDS"))
    used2 = read_stems(os.environ.get("EXCLUDE_SHARDS"))
    stems = sorted(f[:-len(".jsonl.gz")] for f in os.listdir(ST2) if f.endswith(".jsonl.gz"))
    stems = [s for s in stems if s not in held and s not in used2]
    random.Random(11).shuffle(stems)
    print("candidate shards", len(stems), "(held-out excluded", len(held), ", other excluded", len(used2), ")", flush=True)
    seen = seen_hashes(); print("dedup set", len(seen), flush=True)
    stems = stems[:int(target / 1350) + 50]   # enough shards for the target (~1,470 kept rows/shard); no pool.terminate() (hangs with spawn)
    print("processing shards:", len(stems), flush=True)
    tot = collections.Counter(); rows = []; dup = 0
    done = 0
    with get_context("spawn").Pool(workers) as pool:
        for out, c in pool.imap_unordered(one, stems, chunksize=4):
            tot.update(c); done += 1
            if done % 200 == 0: print("shards", done, "rows", len(rows), dict(tot), flush=True)
            for stem, row, t in out:
                h = norm(t)
                if h in seen: dup += 1; continue
                seen.add(h); rows.append((stem, row, t))
            if len(rows) >= target: break
    random.Random(12).shuffle(rows); rows = rows[:target]
    with open(W + "/seed_pages.jsonl", "w", encoding="utf-8") as f:
        for i, (stem, row, t) in enumerate(rows):
            f.write(json.dumps({"i": i, "stem": stem, "row": row, "output": t}, ensure_ascii=False) + "\n")
    print("written", len(rows), "dup", dup, dict(tot), flush=True)
