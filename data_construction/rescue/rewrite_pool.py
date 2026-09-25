"""<rewrite> targets: the RePro 1B rephraser (OLMo-2 based, no thinking mode) over every pool page the two-stage
refiner deleted whose FineWeb-Edu int_edu (score_edu_pool.py) is >= MIN_EDU.

Decoding is SAMPLED as in RePro upstream (lib/repro_olmo.sampling_params: T=1.0, top-p 0.9, 2,048 tokens),
long pages are split into 7,000-character chunks whose paraphrases are rejoined with a space.
No DataMan, no prose-length test, no edu gate on the output; the only filters are sanity checks
(non-empty, not a rephraser artefact per rwclean, word-length ratio within [0.2, 3.0]).
The edu scores are read from the sidecar score_edu_pool.py wrote, so nothing is re-scored.
Output: OUT_DIR/<shard>.jsonl rows {i, text} (default $WORK_DIR/dclm_pipeline/two_stage_pool/rwrepro_olmo_temp1).
usage: rewrite_pool.py <task_id> <n_tasks>
env: WORK_DIR, OUT_DIR, RW_MODEL (the RePro 1B rephraser checkpoint; identifier withheld for anonymity), MIN_EDU
"""
import gzip, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import rwclean
from repro_olmo import make_conversation, extract, strip_rules, CHUNK, sampling_params
from vllm import LLM, SamplingParams

T = os.environ["WORK_DIR"] + "/dclm_pipeline/two_stage_pool"
OUT = os.environ.get("OUT_DIR", T + "/rwrepro_olmo_temp1")
MODEL = os.environ["RW_MODEL"]
MIN_EDU = float(os.environ.get("MIN_EDU", "1"))
MAX_MODEL_LEN, MAX_TOKENS = 4096, 2048
MIN_RATIO, MAX_RATIO = 0.2, 3.0

tid, nt = int(sys.argv[1]), int(sys.argv[2])
os.makedirs(OUT, exist_ok=True)
name = lambda f: OUT + "/" + f.replace(".jsonl.gz", "") + ".jsonl"
shards = sorted(os.listdir(T + "/best"))[tid::nt]
todo = [f for f in shards if not os.path.exists(name(f))]
print("task", tid, "shards", len(shards), "todo", len(todo), flush=True)
if not todo:
    sys.exit(0)
llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9)
sampling = sampling_params   # repro upstream: temperature=1.0, top_p=0.9
tokenizer = llm.get_tokenizer()
t0 = time.time(); npage = nsel = ndirty = 0
for k, f in enumerate(todo):
    a = [json.loads(l)["text"] for l in gzip.open(T + "/input/" + f, "rt", encoding="utf-8")]
    b = [json.loads(l)["text"] for l in gzip.open(T + "/best/" + f, "rt", encoding="utf-8")]
    ep = T + "/edu/" + f.replace(".jsonl.gz", "") + ".jsonl"
    if len(a) != len(b) or not os.path.exists(ep):
        print("SKIP", f, flush=True)
        continue
    edu = {}
    for l in open(ep, encoding="utf-8"):
        o = json.loads(l); edu[o["i"]] = o["int_edu"]
    convs, owner = [], []
    for i, e in edu.items():
        if e < MIN_EDU or b[i].strip() or not a[i].strip():
            continue
        # upstream splits long pages into CHUNK-sized pieces and rejoins the paraphrases
        for c in range(0, len(a[i]), CHUNK):
            convs.append(make_conversation(a[i][c:c + CHUNK], tokenizer)); owner.append(i)
    nsel += len(set(owner))
    res = []
    if convs:
        outs = llm.chat(convs, sampling, use_tqdm=False)
        parts = {}
        for i, o in zip(owner, outs):
            parts.setdefault(i, []).append(extract(o.outputs[0].text))
        for i, v in parts.items():
            t = rwclean.clean(strip_rules(" ".join(v)))
            if not t or rwclean.is_dirty(t):
                ndirty += 1
                continue
            r = len(t.split()) / max(1, len(a[i].split()))
            if r < MIN_RATIO or r > MAX_RATIO:
                continue
            res.append({"i": i, "text": t})
    tmp = name(f) + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fo:
        for x in res:
            fo.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(tmp, name(f))
    npage += len(res)
    if k == 0 or (k + 1) % 5 == 0:
        el = time.time() - t0
        print("task %d shard %d/%d kept %d of %d selected (%d dropped dirty), %.1f min, %.0f pages/GPU-hour"
              % (tid, k + 1, len(todo), npage, nsel, ndirty, el / 60, npage / max(el, 1.0) * 3600), flush=True)
print("task", tid, "DONE shards", len(todo), "pages", npage, "of", nsel, "selected, dirty dropped",
      ndirty, "%.1f min" % ((time.time() - t0) / 60), flush=True)
print("REWRITE_POOL_OK")
