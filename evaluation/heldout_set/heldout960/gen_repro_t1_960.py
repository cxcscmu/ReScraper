"""T=1 RePro-1B paraphrases for the held-out pages that have none in two_stage_pool/rwrepro_olmo_temp1.
Body = the pool rescue generator's generation loop verbatim (same model snapshot, same repro_olmo.make_conversation /
extract / strip_rules / CHUNK / sampling_params = T 1.0, top_p 0.9, max_tokens 2048, same max_model_len 4096,
same rwclean.clean + is_dirty + 0.2-3.0 word-ratio filter). Only the page selection differs: an explicit
list of pool indices (gen_need.jsonl) instead of `int_edu >= 1 and the pool's two-stage refiner deleted it`.
usage: gen_repro_t1_960.py <model_dir> <need.jsonl> <two_stage_pool input.jsonl.gz of the shard> <out.jsonl>"""
import gzip, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "lib"))
import rwclean
from repro_olmo import make_conversation, extract, strip_rules, CHUNK, sampling_params
from vllm import LLM
MODEL, NEED, INP, OUT = sys.argv[1:5]
MAX_MODEL_LEN = 4096
MIN_RATIO, MAX_RATIO = 0.2, 3.0
need = [json.loads(l) for l in open(NEED)]
a = [json.loads(l)["text"] for l in gzip.open(INP, "rt", encoding="utf-8")]
llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9)
sampling = sampling_params
print("sampling", sampling, flush=True)
tokenizer = llm.get_tokenizer()
convs, owner = [], []
for x in need:
    i = x["pool_idx"]
    for c in range(0, len(a[i]), CHUNK):
        convs.append(make_conversation(a[i][c:c + CHUNK], tokenizer)); owner.append(i)
outs = llm.chat(convs, sampling, use_tqdm=False)
parts, raws = {}, {}
for i, o in zip(owner, outs):
    parts.setdefault(i, []).append(extract(o.outputs[0].text)); raws.setdefault(i, []).append(o.outputs[0].text)
with open(OUT, "w", encoding="utf-8") as fo:
    for x in need:
        i = x["pool_idx"]; v = parts.get(i, [])
        t = rwclean.clean(strip_rules(" ".join(v)))
        status = "ok"
        if not t or rwclean.is_dirty(t):
            status = "dirty_or_empty"
        else:
            r = len(t.split()) / max(1, len(a[i].split()))
            if r < MIN_RATIO or r > MAX_RATIO:
                status = "ratio_%.3f" % r
        fo.write(json.dumps(dict(x, i=i, text=t if status == "ok" else "", status=status, n_chunks=len(v),
                                 raw=raws.get(i, [])), ensure_ascii=False) + "\n")
        print(x["gold_idx"], i, status, len(t.split()), "words vs", len(a[i].split()), flush=True)
print("GEN_REPRO_T1_OK")
