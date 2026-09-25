"""Held-out set step 1 (CPU): the 3,161 existing staged held-out rows and the exclusion inputs.
(a) provenance of the 3,161 existing staged held-out rows ($HELDOUT3161_DIR/sft_e2eC_gold_tagged.jsonl): replay the staged
    builder's to_staged_row over $HELDOUT3161_DIR/render_*.jsonl, then render `row` (= simplified-html step3b row, as the
    renderer used it) -> WARC-Record-ID -> raw step3a record = html, url; md5(html) -> Dripper step-1 line (step1 idx);
    re-render check number_lines(webkit_txt(html)) == staged input.
(b) pool join: exact Dripper text -> two_stage_pool/input index of the page's shard; text_decisions edu/action,
    two_stage_pool/edu, rwrepro_olmo_temp1 text, two_stage_pool/best kept flag.
(c) FineWeb-Edu recomputed for every row exactly as heldout960/edu_score960.py (CPU fp32, batches of 64 in table
    order, padding=True, truncation 512) on the joined two_stage_pool/input text (== render drip).
(d) WARC ids + urls of the 50,000-page Dripper SFT source (TWO_STAGE_SFT_SOURCE_GLOB), a superset of the 45,610 pages
    the two-stage refiner (two_stage_refiner) was trained on.
(e) shard sets of the other pool SFT/held-out files (dripper_sft_pool{,2}_{train,heldout}.jsonl).
usage: s1_old_rows.py <outdir>"""
import json, gzip, glob, hashlib, os, sys, re, signal, collections, ast, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib")); sys.path.insert(0, os.path.join(HERE, ".."))
from eval_paths import (SFT_DATA_DIR as D, HELDOUT3161_DIR as GD, TWO_STAGE_POOL_DIR as T, TEXT_DECISIONS_DIR as DEC,
                        POOL_SIMP_DIR as SIMP, POOL_RAW_DIR as RAW, TWO_STAGE_SFT_SOURCE_GLOB)
import e2e_ops as X, editops as E, pool_join as H, render
ST1 = H.ST1
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
md5 = lambda s: hashlib.md5(s.encode("utf-8", "surrogatepass")).hexdigest()
def meta(m):
    if isinstance(m, dict): return m
    try: return ast.literal_eval(m)
    except Exception:
        try: return json.loads(m)
        except Exception: return {}
t0 = time.time()
# ---------- (d) 50k Dripper SFT source WARC ids ----------
WID = re.compile(r"WARC-Record-ID['\"]?\s*:\s*['\"](<urn:uuid:[^>]+>)"); URI = re.compile(r"WARC-Target-URI['\"]?\s*:\s*['\"]([^'\"]*)")
n = 0
with open(OUT + "/two_stage_src_warc.tsv", "w") as fo:
    for f in sorted(glob.glob(TWO_STAGE_SFT_SOURCE_GLOB)):
        for k, l in enumerate(open(f, encoding="utf-8")):
            o = json.loads(l); m = o.get("metadata"); ms = m if isinstance(m, str) else json.dumps(m)
            a = WID.search(ms); b = URI.search(ms)
            fo.write("%s\t%d\t%s\t%s\t%s\n" % (os.path.basename(f), k, a.group(1) if a else "", b.group(1) if b else "", md5(o.get("text") or "")))
            n += 1
print("(d) two-stage SFT source rows", n, "%.0fs" % (time.time() - t0), flush=True)
# ---------- (e) shard sets of the other pool SFT files ----------
SH = re.compile(r'"shard":\s*"([^"]+)"'); sets = {}
for name in ("dripper_sft_pool_train", "dripper_sft_pool_heldout", "dripper_sft_pool2_train", "dripper_sft_pool2_heldout"):
    s = collections.Counter()
    for l in open(D + "/" + name + ".jsonl", encoding="utf-8"):
        m = SH.findall(l); s[m[-1] if m else "?"] += 1
    sets[name] = dict(s); print("(e)", name, "rows", sum(s.values()), "shards", len(s), flush=True)
json.dump(sets, open(OUT + "/other_sft_shards.json", "w"))
# ---------- (a) provenance ----------
rows = [json.loads(l) for f in sorted(glob.glob(GD + "/render_*.jsonl")) for l in open(f, encoding="utf-8")]
prov = []
for r in rows:
    st, kind = X.to_staged_row(r["input"], r["drip"], r["output"])
    if st: prov.append((r, st, kind))
gold = [json.loads(l) for l in open(GD + "/sft_e2eC_gold_tagged.jsonl", encoding="utf-8")]
assert len(prov) == len(gold) == 3161, (len(prov), len(gold))
assert all(p[1]["input"] == g["input"] and p[1]["output"] == g["output"] for p, g in zip(prov, gold)), "replay != staged table"
print("(a) render rows", len(rows), "staged", len(prov), "== staged table rows, byte-identical input+output", flush=True)
by_stem = collections.defaultdict(list)
for k, (r, st, kind) in enumerate(prov): by_stem[r["stem"]].append(k)
rec = [None] * len(prov); html_of = {}
for stem, ks in by_stem.items():
    want = collections.defaultdict(list)
    for k in ks: want[prov[k][0]["row"]].append(k)
    wid = {}
    with gzip.open(f"{SIMP}/{stem}_processed.jsonl.gz", "rt", encoding="utf-8") as g:
        for j, l in enumerate(g):
            if j in want: wid[meta(json.loads(l)["metadata"])["WARC-Record-ID"]] = want[j]
    raw = {}
    with gzip.open(f"{RAW}/{stem}.jsonl.gz", "rt", encoding="utf-8") as g:
        for j, l in enumerate(g):
            o = json.loads(l); m = meta(o["metadata"]); w = m.get("WARC-Record-ID")
            if w in wid: raw[w] = (o["text"], m.get("WARC-Target-URI"), j)
    s1 = collections.defaultdict(list); s1mh = {}
    with open(f"{ST1}/{stem}.jsonl", encoding="utf-8") as g:
        for j, l in enumerate(g):
            o = json.loads(l); h = md5(o.get("input") or ""); s1[h].append(j); s1mh[j] = bool((o.get("main_html") or "").strip())
    for w, klist in wid.items():
        for k in klist:
            html, url, rawi = raw.get(w, (None, None, None))
            h = md5(html) if html is not None else None
            idx = s1.get(h, []) if h else []
            rec[k] = {"gold_idx": k, "stem": stem, "simp_row": prov[k][0]["row"], "render_i": prov[k][0]["i"], "render_tag": prov[k][0]["tag"],
                      "staged_kind": prov[k][2], "warc_id": w, "url": url, "raw_row": rawi, "in_md5": h,
                      "step1_idx": idx[0] if idx else None, "step1_n_same_md5": len(idx), "step1_has_main_html": s1mh.get(idx[0]) if idx else None}
            if html is not None: html_of[k] = html
    print("(a)", stem, "rows", len(ks), "warc", len(wid), "raw found", len(raw), flush=True)
miss = [k for k in range(len(prov)) if rec[k] is None]
for k in miss: rec[k] = {"gold_idx": k, "stem": prov[k][0]["stem"], "simp_row": prov[k][0]["row"], "warc_id": None}
print("(a) rows without a simp->warc->raw record:", len(miss), flush=True)
# re-render check (lib/render.py webkit_txt = the renderer of every ReScraper input)
render._load_converter()       # import + patch the renderer once, before the worker processes fork
webkit_txt = render.webkit_txt
class TO(Exception): pass
def _init(): signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))
def render_page(h, sec=90):
    signal.alarm(sec)
    try: return webkit_txt(h)
    except Exception: return None
    finally: signal.alarm(0)
from multiprocessing import Pool
ks = sorted(html_of)
with Pool(int(os.environ.get("PROCS", "30")), initializer=_init) as pool:
    rend = pool.map(render_page, [html_of[k] for k in ks], chunksize=4)
C = collections.Counter()
for k, t in zip(ks, rend):
    ok = t is not None and E.number_lines(t) == gold[k]["input"]
    rec[k]["rerender_eq_input"] = ok; C[ok] += 1
print("(a) re-render == staged input:", dict(C), "%.0fs" % (time.time() - t0), flush=True)
with open(OUT + "/html_old.jsonl", "w", encoding="utf-8") as f:
    for k in ks: f.write(json.dumps({"gold_idx": k, "html": html_of[k]}, ensure_ascii=False) + "\n")
# ---------- (b) pool join (as heldout960/join_pool960.py) ----------
for stem, kk in by_stem.items():
    inp = [json.loads(l)["text"] for l in gzip.open(f"{T}/input/{stem}.jsonl.gz", "rt", encoding="utf-8")]
    best = [json.loads(l)["text"] for l in gzip.open(f"{T}/best/{stem}.jsonl.gz", "rt", encoding="utf-8")]
    pos = collections.defaultdict(list)
    for j, t in enumerate(inp): pos[t].append(j)
    td = {o["row"]: o for o in map(json.loads, open(f"{DEC}/{stem}.jsonl"))}
    te = {o["i"]: o for o in map(json.loads, open(f"{T}/edu/{stem}.jsonl"))}
    rw = {}
    for o in map(json.loads, open(f"{T}/rwrepro_olmo_temp1/{stem}.jsonl", encoding="utf-8")):
        t = (o.get("text") or "").strip()
        if t: rw[int(o["i"])] = t
    for k in kk:
        js = pos.get(prov[k][0]["drip"], [])
        rec[k].update({"pool_idx": js, "td_edu": [td[j]["edu"] if j in td else None for j in js], "td_action": [td[j]["action"] if j in td else None for j in js],
                       "pool_edu": [te[j]["edu"] if j in te else None for j in js], "pool_int_edu": [te[j]["int_edu"] if j in te else None for j in js],
                       "pool_student_kept": [bool(best[j].strip()) for j in js], "rw_t1": [rw.get(j) for j in js]})
    print("(b)", stem, "pool input rows", len(inp), "joined", sum(1 for k in kk if rec[k]["pool_idx"]), "/", len(kk), flush=True)
# ---------- (c) edu recompute, as heldout960/edu_score960.py ----------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(int(os.environ.get("EDU_THREADS", "16")))
etok = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
emod = AutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier").eval()
@torch.no_grad()
def escore(ts):
    out = []
    for i in range(0, len(ts), 64):
        e = etok(ts[i:i+64], return_tensors="pt", padding=True, truncation=True, max_length=512)
        out += emod(**e).logits.squeeze(-1).float().cpu().tolist()
    return out
es = escore([prov[k][0]["drip"] for k in range(len(prov))])
for k, s in enumerate(es): rec[k]["edu_raw_recomputed"] = s; rec[k]["edu_recomputed"] = round(float(s), 3)
print("(c) edu recomputed", len(es), "%.0fs" % (time.time() - t0), flush=True)
with open(OUT + "/old_prov.jsonl", "w", encoding="utf-8") as f:
    for r in rec: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("STEP1_OLD_OK", flush=True)
