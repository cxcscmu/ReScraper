"""ProX-C (chunk-level only) on a list of pages (page-level runner, e.g. the resiliparse text of held-out pages).
Settings copied from proxc_pool.py: trunc_text (<=1500 tokens, [NNN]-prefixed lines), prompt format "plain" (the format
every rank of the pool run selected), SamplingParams(temperature=0, top_p=0.9, max_tokens=256), max_model_len 2048 with
over-long chunks skipped, ProX merge_chunks, vendored ProX executor (lib/prox_chunk_utils.py) with threshold_1=0.0,
threshold_2=0.95, error_op=2. An execute failure drops the page, as in proxc_pool.py (it `continue`s without writing).
usage: proxc_pages.py <pages.jsonl {k,text}> <out.jsonl {k,proxc,outcome,program}>
env: PROXC_MODEL (default gair-prox/web-chunk-refining-lm)
"""
import collections, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from prox_chunk_utils import execute_meta_operations
from vllm import LLM, SamplingParams
MODEL = os.environ.get("PROXC_MODEL", "gair-prox/web-chunk-refining-lm")
MAX_LEN, MAX_NEW, CHUNK_TOKENS = 2048, 256, 1500
llm = LLM(model=MODEL, dtype="auto", max_model_len=MAX_LEN, gpu_memory_utilization=0.85, enable_prefix_caching=False)
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=MAX_NEW)
def trunc_text(text, max_token=CHUNK_TOKENS, max_digits=3):
    lines = text.split("\n")
    norm = ["[%0*d]%s" % (max_digits, i, l) for i, l in enumerate(lines)]
    counts = [len(x) for x in tok(norm)["input_ids"]]
    chunks, cur, cur_n = [], [], 0
    for nl, c in zip(norm, counts):
        if cur_n + c <= max_token:
            cur.append(nl); cur_n += c
        else:
            if cur: chunks.append("\n".join(cur))
            cur = [nl]; cur_n = c
            if c > max_token:
                chunks.append("\n".join(cur)); cur = []; cur_n = 0
    if cur: chunks.append("\n".join(cur))
    return chunks
def wrap_plain(chunk): return "[doc]\n%s\n[/doc]" % chunk
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
prompts, lens, S = [], [], collections.Counter()
for r in rows:
    keep = []
    for c in trunc_text(r["text"]):
        p = wrap_plain(c)
        if len(tok(p)["input_ids"]) > MAX_LEN - MAX_NEW: S["chunk_too_long_skipped"] += 1; continue
        keep.append(p)
    lens.append(len(keep)); prompts.extend(keep)
print("pages", len(rows), "chunks", len(prompts), flush=True)
outs = [o.outputs[0].text.strip(" ") for o in llm.generate(prompts, sp, use_tqdm=True)]
i = 0
with open(sys.argv[2], "w", encoding="utf-8") as fo:
    for r, n in zip(rows, lens):
        prog = "\n".join(outs[i:i + n]); i += n
        try:
            new = execute_meta_operations(text=r["text"], operations=prog, threshold_1=0.0, threshold_2=0.95, error_op=2)
            oc = "blanked" if not new or not new.strip() else ("unchanged" if new == r["text"] else "edited")
        except Exception:
            new, oc = "", "execute_fail"
        S[oc] += 1
        fo.write(json.dumps({"k": r["k"], "proxc": new or "", "outcome": oc, "program": prog[-2000:]}, ensure_ascii=False) + "\n")
print(dict(S)); print("PROXC_DONE", flush=True)
