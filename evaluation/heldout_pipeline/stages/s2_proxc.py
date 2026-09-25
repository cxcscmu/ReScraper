"""Stage 2c (GPU): ProX-C (gair-prox/web-chunk-refining-lm) on the resiliparse text of every page.

Same settings as the pool-scale ProX-C run (baselines/): trunc_text (<=1500 tokens, [NNN]-prefixed lines), prompt format
"plain" ("[doc]\\n...\\n[/doc]"), SamplingParams(temperature=0, top_p=0.9, max_tokens=256), max_model_len 2048 with
over-long chunks skipped, LLM(dtype=auto, gpu_memory_utilization=0.85, enable_prefix_caching=False), vendored ProX
executor lib/prox_chunk_utils.py execute_meta_operations(threshold_1=0.0, threshold_2=0.95, error_op=2). Input =
page["resiliparse"] keyed by gid; pages go in chunks of PROXC_CHUNK (default 1000) per llm.generate call, each chunk
written on completion (a requeued job resumes).
Output <work>/sys_proxc.jsonl {gid, text|null, status (outcome), ops, w_in, w_out, program (last 2000 chars)}
usage: s2_proxc.py <page_table>
"""
import collections, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib")); sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "lib"))
from common import *
from prox_chunk_utils import execute_meta_operations

PT = sys.argv[1]; W = workdir(PT); PARTS = W + "/proxc_parts"; os.makedirs(PARTS, exist_ok=True)
CHUNK = int(os.environ.get("PROXC_CHUNK", "1000"))
MODEL = os.environ.get("PROXC_MODEL", "gair-prox/web-chunk-refining-lm")
MAX_LEN, MAX_NEW, CHUNK_TOKENS = 2048, 256, 1500
pages = load_pages(PT, ["gid", "resiliparse"])
chunks = [pages[i:i + CHUNK] for i in range(0, len(pages), CHUNK)]
todo = [c for c in range(len(chunks)) if not os.path.exists("%s/part%03d.jsonl" % (PARTS, c))]
print("pages %d chunks %d todo %s" % (len(pages), len(chunks), todo), flush=True)
if todo:
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="auto", max_model_len=MAX_LEN, gpu_memory_utilization=0.85, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=MAX_NEW)

def trunc_text(text, max_token=CHUNK_TOKENS, max_digits=3):      # verbatim from the ProX-C runner
    lines = text.split("\n")
    norm = ["[%0*d]%s" % (max_digits, i, l) for i, l in enumerate(lines)]
    counts = [len(x) for x in tok(norm)["input_ids"]]
    chunks, cur, cur_n = [], [], 0
    for nl, c in zip(norm, counts):
        if cur_n + c <= max_token:
            cur.append(nl); cur_n += c
        else:
            if cur:
                chunks.append("\n".join(cur))
            cur = [nl]; cur_n = c
            if c > max_token:
                chunks.append("\n".join(cur)); cur = []; cur_n = 0
    if cur:
        chunks.append("\n".join(cur))
    return chunks

def wrap_plain(chunk):
    return "[doc]\n%s\n[/doc]" % chunk

CALL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
S = collections.Counter()
for c in todo:
    prompts, lens = [], []
    for p in chunks[c]:
        d = p["resiliparse"] or ""
        if not d.strip():
            lens.append(0); S["empty_input"] += 1; continue
        keep = []
        for ch in trunc_text(d):
            pr = wrap_plain(ch)
            if len(tok(pr)["input_ids"]) > MAX_LEN - MAX_NEW:
                S["chunk_too_long_skipped"] += 1; continue
            keep.append(pr)
        lens.append(len(keep)); prompts.extend(keep)
    outs = [o.outputs[0].text.strip(" ") for o in llm.generate(prompts, sp, use_tqdm=False)] if prompts else []
    i = 0; recs = []
    for p, n in zip(chunks[c], lens):
        d = p["resiliparse"] or ""
        prog = "\n".join(outs[i:i + n]); i += n
        names = [m.group(1) for l in prog.split("\n") for m in [CALL.match(l)] if m]
        if not d.strip():
            recs.append({"gid": p["gid"], "text": None, "status": "no_input", "ops": [], "w_in": 0, "w_out": 0, "program": ""}); continue
        try:
            new = execute_meta_operations(text=d, operations=prog, threshold_1=0.0, threshold_2=0.95, error_op=2)
        except Exception:
            new = None
        if new is None:
            oc = "execute failed"
        elif not new.strip():
            oc = "page blanked"
        elif new == d:
            oc = "unchanged"
        else:
            s = set(names)
            oc = "edited: " + ("lines + normalize" if {"remove_lines", "normalize"} <= s else
                               "lines only" if "remove_lines" in s else
                               "normalize only" if "normalize" in s else "other")
        recs.append({"gid": p["gid"], "text": new if (new and new.strip()) else None, "status": oc, "ops": names,
                     "w_in": len(d.split()), "w_out": len(new.split()) if new else 0, "program": prog[-2000:]})
    assert i == len(outs)
    write_jsonl_atomic("%s/part%03d.jsonl" % (PARTS, c), recs)
    print("chunk %d/%d: %d pages, %d prompts" % (c + 1, len(chunks), len(recs), len(prompts)), dict(S), flush=True)
allr = []
for c in range(len(chunks)):
    allr += list(read_jsonl("%s/part%03d.jsonl" % (PARTS, c)))
assert [r["gid"] for r in allr] == list(range(len(pages)))
write_jsonl_atomic(W + "/sys_proxc.jsonl", allr)
V = {"outcomes": dict(collections.Counter(r["status"] for r in allr)),
     "words_kept_pct": 100.0 * sum(r["w_out"] for r in allr) / max(1, sum(r["w_in"] for r in allr))}
json.dump(V, open(W + "/sys_proxc.validation.json", "w"), indent=1)
print(json.dumps(V, indent=1)); print("PROXC_OK", flush=True)
