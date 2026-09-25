"""Pool-scale inference of the ReScraper model (the released corpus), reading the raw HTML directly.

Input universe = Dripper step 1 over the pool ($DRIPPER_STEP1_DIR/<stem>.jsonl, lib/pool_join.ST1): every record
with a non-empty raw HTML `input` and a non-empty `main_html`. The raw HTML is exactly what the training inputs
were rendered from, so no content alignment is needed.

Per shard: render the raw HTML with the training renderer (webkit_txt, 60 s timeout, pages under 20 characters
dropped), number the lines (<lid:n>), skip prompts over 32,768 - 3,072 = 29,696 tokens, generate with
T=1.0 / top-p 1.0 / 3,072 new tokens / thinking disabled (vLLM, bf16, prefix caching), execute the program with
lib/e2e_ops.body_from_prediction_dfirst, drop <delete> and empty pages, and write {"text", "e2e_tag"} rows to
$INFER_OUT_DIR/<stem>_processed.jsonl.gz (atomic, resumable per shard).
usage: infer_pool.py <rank> <world>
env: INFER_MODEL_PATH (the Stage-2 checkpoint), INFER_SYSTEM_PROMPT_FILE (prompts/student_system_stage2.txt),
     INFER_OUT_DIR, INFER_STEMS_FILE (optional: explicit list of shard stems, e.g. to backfill missing shards),
     LIMIT_FILES, RENDER_PROCS, DRIPPER_STEP1_DIR
"""
import os, sys, json, gzip, re, time, signal, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import e2e_ops as X, editops as E, pool_join as H
OUT = os.environ["INFER_OUT_DIR"]; MODEL = os.environ["INFER_MODEL_PATH"]
SP = open(os.environ["INFER_SYSTEM_PROMPT_FILE"], encoding="utf-8").read().strip()
MAX_LEN = 32768; MAX_NEW = 3072; MAX_PROMPT = MAX_LEN - MAX_NEW
rank, world = int(sys.argv[1]), int(sys.argv[2])
if os.environ.get("INFER_STEMS_FILE"):   # backfill: explicit stem list
    stems = [l.strip() for l in open(os.environ["INFER_STEMS_FILE"], encoding="utf-8") if l.strip()][rank::world]
else:
    stems = sorted(f[:-6] for f in os.listdir(H.ST1) if f.endswith(".jsonl"))[rank::world]
lim = int(os.environ.get("LIMIT_FILES", "0")); stems = stems[:lim] if lim else stems
os.makedirs(OUT, exist_ok=True)
todo = [s for s in stems if not os.path.exists(os.path.join(OUT, s + "_processed.jsonl.gz"))]
print(f"rank {rank}/{world} shards {len(stems)} todo {len(todo)}", flush=True)
if not todo: sys.exit(0)

import dripper.process.simplify_html as sh
sh.tags_to_remove = {"head", "style", "script", "noscript", "link", "meta", "iframe", "frame"}; sh.ATTR_PATTERNS_TO_REMOVE = set()
from webpage_converter.convert import convert_html_to_structured_data
from bs4 import BeautifulSoup
def webkit_txt(html):
    raw = convert_html_to_structured_data(html, output_format="txt"); text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    text = re.sub(r"\|+", " ", text); text = re.sub(r"^[\s\-\|]+$", "", text, flags=re.M); text = re.sub(r"\\([\$\*\[\]_`#])", r"\1", text)
    return "\n".join(l for l in (re.sub(r"\s+", " ", x).strip() for x in text.split("\n")) if l)
class TO(Exception): pass
def _init(): signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))
def render(h, sec=60):
    signal.alarm(sec)
    try: return webkit_txt(h)
    except Exception: return None
    finally: signal.alarm(0)
from multiprocessing import Pool
pool = Pool(int(os.environ.get("RENDER_PROCS", "10")), initializer=_init)

from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=0.9, enable_prefix_caching=True)
tok = llm.get_tokenizer(); sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=MAX_NEW)
def prompt_of(ninp):
    msgs = [{"role": "system", "content": SP}, {"role": "user", "content": ninp}]
    try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
S = collections.Counter()
for stem in todo:
    t0 = time.time(); raws = []
    with open(f"{H.ST1}/{stem}.jsonl", encoding="utf-8") as g:
        for line in g:
            try: r1 = json.loads(line)
            except ValueError: S["bad_step1_json"] += 1; continue
            raw = r1.get("input") or ""
            # Universe = the pages the extractor actually produced text for, i.e. step1 records with a non-empty
            # main_html. Measured on 6 shards: 10,446 such records vs 10,354 step2 rows (+0.9%), while the
            # length-filtered pool has -14%. So this matches the page set of the baseline corpora within 1%.
            if not raw.strip(): S["empty_step1_input"] += 1; continue
            if not (r1.get("main_html") or "").strip(): S["no_main_html"] += 1; continue
            raws.append(raw)
    rendered = pool.map(render, raws, chunksize=16)
    prompts, ninps = [], []
    for txt in rendered:
        if not txt or len(txt) < 20: S["render_fail_or_empty"] += 1; continue
        ninp = E.number_lines(txt); p = prompt_of(ninp)
        if len(tok.encode(p)) > MAX_PROMPT: S["too_long"] += 1; continue
        prompts.append(p); ninps.append(ninp)
    outs = llm.generate(prompts, sp, use_tqdm=False) if prompts else []
    dst = os.path.join(OUT, stem + "_processed.jsonl.gz"); tmp = dst + ".tmp%d" % os.getpid(); n_out = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as g:
        for ninp, o in zip(ninps, outs):
            try: tag, text = X.body_from_prediction_dfirst(ninp, o.outputs[0].text)
            except Exception: S["parse_fail"] += 1; continue
            S["tag" + str(tag)] += 1
            if tag == "<delete>" or not (text or "").strip(): continue
            g.write(json.dumps({"text": text, "e2e_tag": tag}, ensure_ascii=False) + "\n"); n_out += 1
    os.replace(tmp, dst); S["pages"] += len(raws); S["written"] += n_out
    print(f"rank {rank} {stem}: pages {len(raws)} kept {n_out} in {time.time()-t0:.0f}s | {dict(S)}", flush=True)
pool.close(); pool.join()
print(f"rank {rank} DONE {dict(S)}", flush=True)
