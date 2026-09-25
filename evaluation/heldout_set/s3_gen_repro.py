"""Held-out set step 3c (1 GPU): T=1 RePro-1B paraphrases for the pages that need one and have none in
two_stage_pool/rwrepro_olmo_temp1.
Body = heldout960/gen_repro_t1_960.py (itself the pool rescue generator's loop verbatim): same model snapshot,
repro_olmo.make_conversation / extract / strip_rules / CHUNK / sampling_params (T 1.0, top_p 0.9, max_tokens 2048),
max_model_len 4096, rwclean.clean + is_dirty + 0.2-3.0 word-ratio filter. Only difference: the page text comes from the
need file (field "text" = the Dripper text, i.e. the two_stage_pool/input text) instead of indexing the pool input by
pool_idx. Resumable: exits if <out.jsonl> exists (written atomically).
usage: s3_gen_repro.py <model_dir> <need.jsonl> <out.jsonl>"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import rwclean
from repro_olmo import make_conversation, extract, strip_rules, CHUNK, sampling_params
MODEL, NEED, OUT = sys.argv[1:4]
if os.path.exists(OUT):
    print("exists, skip", OUT); print("GEN_REPRO_T1_OK"); raise SystemExit(0)
from vllm import LLM
MAX_MODEL_LEN = 4096
MIN_RATIO, MAX_RATIO = 0.2, 3.0
need = [json.loads(l) for l in open(NEED)]
if not need:
    open(OUT, "w").close(); print("nothing to generate"); print("GEN_REPRO_T1_OK"); raise SystemExit(0)
llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9)
sampling = sampling_params
print("sampling", sampling, flush=True)
tokenizer = llm.get_tokenizer()
convs, owner = [], []
for x in need:
    i, a = x["key"], x["text"]
    for c in range(0, len(a), CHUNK):
        convs.append(make_conversation(a[c:c + CHUNK], tokenizer)); owner.append(i)
outs = llm.chat(convs, sampling, use_tqdm=False)
parts, raws = {}, {}
for i, o in zip(owner, outs):
    parts.setdefault(i, []).append(extract(o.outputs[0].text)); raws.setdefault(i, []).append(o.outputs[0].text)
with open(OUT + ".tmp", "w", encoding="utf-8") as fo:
    for x in need:
        i, a = x["key"], x["text"]; v = parts.get(i, [])
        t = rwclean.clean(strip_rules(" ".join(v)))
        status = "ok"
        if not t or rwclean.is_dirty(t):
            status = "dirty_or_empty"
        else:
            r = len(t.split()) / max(1, len(a.split()))
            if r < MIN_RATIO or r > MAX_RATIO:
                status = "ratio_%.3f" % r
        fo.write(json.dumps(dict({k: v2 for k, v2 in x.items() if k != "text"}, text=t if status == "ok" else "", status=status,
                                 n_chunks=len(v), raw=raws.get(i, [])), ensure_ascii=False) + "\n")
        print(i, status, len(t.split()), "words vs", len(a.split()), flush=True)
os.replace(OUT + ".tmp", OUT)
print("GEN_REPRO_T1_OK")
