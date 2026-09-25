"""Cumulative distinct n-grams over the same 376 pool shards for every corpus (Figure 6, right).

Exact distinct counting over 64-bit keys (n<=4: the token ids packed into one uint64, collision-free; n>4: chained
splitmix64 hash h <- splitmix64(h ^ t_i)), the union kept as a sorted uint64 array (searchsorted + np.insert merge),
no sketching. Shards: every stem of stems376.txt (the 376 pool shards for which all four corpora exist; the pool-scale
ProX-C run finished on 376 of 10,318 shards), in the fixed order random.Random(0).shuffle(sorted(stems)), same for every
corpus. Tokenizer GPT-NeoX-20B (add_special_tokens=False); n-grams within documents only; empty / whitespace-only docs
dropped. Also per shard: the sum over documents of each document's own distinct n-grams ("per_doc_distinct"), so repeats
split into within-document repeats (occurrences - per_doc_distinct) and cross-document repeats (per_doc_distinct - union
distinct).
Corpora (all before deduplication; paths overridable by env):
  rescraper        $RESCRAPER_CORPUS_DIR/<stem>_processed.jsonl.gz, field text (released corpus after the post-filter)
  ultrax           $ULTRAX_RERUN_DIR/post/<stem>_processed.parquet, column cleaned (UltraX over the raw resiliparse text)
  proxc            $PROXC_POOL_DIR/<stem>_processed.jsonl.gz, field text (ProX-C over the raw resiliparse text)
  refinedweb_rule  $RW_RULE_POOL_DIR/<stem>_processed.jsonl.gz, field text (resiliparse + DCLM RefinedWeb rules)
  resiliparse_raw  $ULTRAX_RERUN_DIR/post/<stem>_processed.parquet, column original (not plotted)
  fineweb_rule     $DIV_DIR/fw/<stem>.jsonl.gz from fw_scale.py (not plotted)
usage: ngram_scale.py <corpus> [<stems file>] [<out tag>]   -> $DIV_DIR/results/<corpus>[_<tag>].json
"""
import gzip, itertools, json, os, random, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("WORK_DIR") or sys.exit("set WORK_DIR")
DIV = os.environ.get("DIV_DIR") or os.path.join(os.environ.get("EVAL_DIR") or os.path.join(WORK, "eval"), "diversity")
NS = (3, 5, 7)
REL = os.environ.get("RESCRAPER_CORPUS_DIR") or os.path.join(WORK, "dclm_pipeline", "rescraper_corpus", "text_clean")
UX = os.environ.get("ULTRAX_RERUN_DIR") or os.path.join(WORK, "dclm_pipeline", "resiliparse_raw_ultrax")
PROXC = os.environ.get("PROXC_POOL_DIR") or os.path.join(WORK, "dclm_pipeline", "prox_c", "text")
B_RW = os.environ.get("RW_RULE_POOL_DIR") or os.path.join(WORK, "dclm_pipeline", "resiliparse", "step3b_post_lang",
                                                           "dclm_baseline_refinedweb_post_lang", "processed_data")
CORPORA = {  # corpus -> (format, path template, field, provenance)
    "rescraper": ("jsonl", REL + "/{stem}_processed.jsonl.gz", "text",
                  "released ReScraper corpus after the post-filter, before BFF dedup (<stem>_processed.jsonl.gz)"),
    "ultrax": ("parquet", UX + "/post/{stem}_processed.parquet", "cleaned",
               "UltraX over the RAW resiliparse extraction, column cleaned, non-empty (no length/language/rule filter; before dedup)"),
    "proxc": ("jsonl", PROXC + "/{stem}_processed.jsonl.gz", "text",
              "ProX-C web-chunk-refining-lm over the raw resiliparse extraction of the same pool (no length/LID/rule filter, no dedup; the run covers 376 shards)"),
    "refinedweb_rule": ("jsonl", B_RW + "/{stem}_processed.jsonl.gz", "text",
                        "resiliparse + DCLM lang filter + full RefinedWeb rules, before BFF dedup"),
    "resiliparse_raw": ("parquet", UX + "/post/{stem}_processed.parquet", "original",
                        "raw resiliparse text (= UltraX input), column original of the UltraX parquets, non-empty"),
    "fineweb_rule": ("jsonl", DIV + "/fw/{stem}.jsonl.gz", "text", "FineWeb-rule chain (heldout_pipeline s2_fw_rule.py code, fw_scale.py) on the raw resiliparse text of the same shards"),
}


def load_texts(corpus, stem):
    fmt, tmpl, field, _ = CORPORA[corpus]
    path = tmpl.format(stem=stem)
    if fmt == "parquet":
        import pyarrow.parquet as pq
        texts = pq.read_table(path, columns=[field]).column(field).to_pylist()
    else:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8") as f:
            texts = [json.loads(l)[field] for l in f if l.strip()]
    return [t for t in texts if t and t.strip()]


M1, M2C, M3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def splitmix64(z):
    z = z + M1
    z = (z ^ (z >> np.uint64(30))) * M2C
    z = (z ^ (z >> np.uint64(27))) * M3
    return z ^ (z >> np.uint64(31))


def ngram_keys(tok, lens, n):
    """uint64 key of every within-document n-gram and the index of its document."""
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    docid = np.repeat(np.arange(len(lens)), lens)
    pos = np.arange(len(tok)) - np.repeat(offs, lens)
    starts = np.nonzero(pos <= np.repeat(lens, lens) - n)[0]
    k = np.zeros(len(starts), dtype=np.uint64)
    if n <= 4:
        for i in range(n):
            k = (k << np.uint64(16)) | tok[starts + i]
    else:
        for i in range(n):
            k = splitmix64(k ^ tok[starts + i])
    return k, docid[starts]


def run(corpus, stems_file, tag):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tk.model_max_length = 10 ** 12
    stems = sorted(l.strip() for l in open(stems_file) if l.strip())
    order = list(stems)
    random.Random(0).shuffle(order)
    union = {n: np.zeros(0, dtype=np.uint64) for n in NS}
    per_shard = []
    cum = {"docs": [], "tokens": [], "distinct": {str(n): [] for n in NS}, "ngram_occurrences": {str(n): [] for n in NS},
           "per_doc_distinct": {str(n): [] for n in NS}}
    cd = ct = 0
    co = {n: 0 for n in NS}; cp = {n: 0 for n in NS}
    t0 = time.time()
    with np.errstate(over="ignore"):
        for j, stem in enumerate(order):
            texts = load_texts(corpus, stem)
            ids = tk(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"] if texts else []
            lens = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
            tok = np.fromiter(itertools.chain.from_iterable(ids), dtype=np.uint64, count=int(lens.sum()))
            rec = {"stem": stem, "docs": len(texts), "tokens": int(lens.sum()), "distinct_in_shard": {}, "ngram_occurrences": {}}
            cd += len(texts); ct += int(lens.sum())
            for n in NS:
                if len(tok):
                    k, d = ngram_keys(tok, lens, n)
                else:
                    k, d = np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)
                pd = int(len(np.unique(splitmix64(k ^ splitmix64(d.astype(np.uint64) + np.uint64(0x5151)))))) if len(k) else 0
                u = np.unique(k)
                rec["distinct_in_shard"][str(n)] = int(len(u)); rec["ngram_occurrences"][str(n)] = int(len(k))
                U = union[n]
                idx = np.searchsorted(U, u)
                hit = np.zeros(len(u), dtype=bool)
                ok = idx < len(U)
                hit[ok] = U[idx[ok]] == u[ok]
                union[n] = np.insert(U, idx[~hit], u[~hit])
                co[n] += len(k); cp[n] += pd
                cum["distinct"][str(n)].append(int(len(union[n])))
                cum["ngram_occurrences"][str(n)].append(int(co[n]))
                cum["per_doc_distinct"][str(n)].append(int(cp[n]))
            cum["docs"].append(cd); cum["tokens"].append(ct)
            per_shard.append(rec)
            if j % 10 == 0 or j == len(order) - 1:
                print("%s %3d/%d docs=%d tok=%d | cum tok=%d distinct %s | %.0fs" % (
                    corpus, j + 1, len(order), rec["docs"], rec["tokens"], ct,
                    " ".join("%d:%d" % (n, len(union[n])) for n in NS), time.time() - t0), flush=True)
    for n in NS:
        assert len(union[n]) < 2 or bool(np.all(union[n][1:] > union[n][:-1]))
    out = {"corpus": corpus, "source": CORPORA[corpus][3], "field": CORPORA[corpus][2], "order": order, "stems_file": stems_file,
           "ns": list(NS), "per_shard": per_shard, "cumulative": cum, "secs": round(time.time() - t0, 1)}
    os.makedirs(DIV + "/results", exist_ok=True)
    name = corpus + ("_" + tag if tag else "")
    tmp = DIV + "/results/%s.json.tmp" % name
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, DIV + "/results/%s.json" % name)
    print("DONE", name, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else HERE + "/stems376.txt", sys.argv[3] if len(sys.argv) > 3 else "")
