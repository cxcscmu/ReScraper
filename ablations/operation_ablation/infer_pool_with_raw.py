"""Operation-ablation copy of inference/infer_pool.py (the script that produced the released corpus). Decoding,
rendering, prompt, executor and the written text rows are unchanged line for line. What this copy adds:
  1. INFER_STEMS_FILE (optional): explicit stem list for backfill, exactly like infer_e2e_text_s1_bf.py.
  2. The two POISON stems (renderer hangs forever, signal.alarm cannot interrupt C code) are always skipped;
     the released corpus lacks them too (10,318 of 10,320 shards).
  3. Per shard it ALSO writes $ABLATION_RAW_DIR/<stem>_raw.jsonl.gz, one row per step1 record, sorted by idx:
       key    stem, idx (0-based line number in step1 <stem>.jsonl), in_md5 (md5 of the raw html = the key
              to the raw-pool WARC-Record-ID; step1 rows carry no url)
       skipped pages: skip = bad_step1_json | empty_step1_input | no_main_html | render_fail_or_empty | too_long
       generated pages: gen (raw generation), finish, n_gen_tok, tag (executor tag, None on parse_fail),
              final (executor text = what the released script writes), extracted (text after the model's own
              <extract> ops only, parsed exactly as body_from_prediction_dfirst parses them), in_full (row written
              to text/), and ninp (numbered input) only when ABLATION_SAVE_NINP=1 (smoke test).
  4. Resume: a shard is redone unless BOTH outputs exist; raw is renamed into place before text.
usage: infer_pool_with_raw.py <rank> <world>
env: INFER_MODEL_PATH, INFER_SYSTEM_PROMPT_FILE, INFER_OUT_DIR, ABLATION_RAW_DIR, INFER_STEMS_FILE, ABLATION_SAVE_NINP,
     LIMIT_FILES, RENDER_PROCS
"""
import os, sys, json, gzip, re, time, signal, collections, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import rescraper_ops as X, editops as E, pool_join as H
OUT = os.environ["INFER_OUT_DIR"]; MODEL = os.environ["INFER_MODEL_PATH"]
SP = open(os.environ["INFER_SYSTEM_PROMPT_FILE"], encoding="utf-8").read().strip()
RAWD = os.environ["ABLATION_RAW_DIR"]; SAVE_NINP = os.environ.get("ABLATION_SAVE_NINP", "") == "1"
POISON = frozenset(("CC-MAIN-20200807231820-20200808021820-00542", "CC-MAIN-20200812053726-20200812083726-00050"))
MAX_LEN = 32768; MAX_NEW = 3072; MAX_PROMPT = MAX_LEN - MAX_NEW
rank, world = int(sys.argv[1]), int(sys.argv[2])
if os.environ.get("INFER_STEMS_FILE"):
    stems = [l.strip() for l in open(os.environ["INFER_STEMS_FILE"], encoding="utf-8") if l.strip()][rank::world]  # backfill: explicit stem list
else:
    stems = sorted(f[:-6] for f in os.listdir(H.ST1) if f.endswith(".jsonl"))[rank::world]
lim = int(os.environ.get("LIMIT_FILES", "0")); stems = stems[:lim] if lim else stems
os.makedirs(OUT, exist_ok=True); os.makedirs(RAWD, exist_ok=True)
def _done(s): return os.path.exists(os.path.join(OUT, s + "_processed.jsonl.gz")) and os.path.exists(os.path.join(RAWD, s + "_raw.jsonl.gz"))
n_poison = sum(s in POISON for s in stems)
todo = [s for s in stems if s not in POISON and not _done(s)]
print(f"rank {rank}/{world} shards {len(stems)} poison_skipped {n_poison} todo {len(todo)} ops_tolerant={X.OPS_TOLERANT} save_ninp={SAVE_NINP}", flush=True)
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

def extracted_of(ninp, pred):
    """Text after the model's own <extract> ops only - the ops1 that body_from_prediction_dfirst would parse:
    decision-first reading when line 1 is a bare decision tag, else the staged fallback reading."""
    lines = pred.strip().split("\n")
    d = lines[0].strip() if lines else ""
    ops1 = X.parse_staged("\n".join(lines[1:]))[0] if d in X.DFIRST_TAGS else X.parse_staged(pred)[0]
    return E.apply_ops(ninp, ops1)
def md5(s): return hashlib.md5(s.encode("utf-8", "surrogatepass")).hexdigest()

from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=0.9, enable_prefix_caching=True)
tok = llm.get_tokenizer(); sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=MAX_NEW)
def prompt_of(ninp):
    msgs = [{"role": "system", "content": SP}, {"role": "user", "content": ninp}]
    try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
S = collections.Counter()
for stem in todo:
    t0 = time.time(); raws = []; keys = []; recs = []
    with open(f"{H.ST1}/{stem}.jsonl", encoding="utf-8") as g:
        for idx, line in enumerate(g):
            try: r1 = json.loads(line)
            except ValueError: S["bad_step1_json"] += 1; recs.append({"stem": stem, "idx": idx, "skip": "bad_step1_json"}); continue
            raw = r1.get("input") or ""
            # Universe = the pages the extractor actually produced text for, i.e. step1 records with a non-empty
            # main_html. Measured on 6 shards: 10,446 such records vs 10,354 step2 rows (+0.9%), while the
            # length-filtered pool has -14%. So this matches the page set of the published corpora within 1%.
            if not raw.strip(): S["empty_step1_input"] += 1; recs.append({"stem": stem, "idx": idx, "skip": "empty_step1_input"}); continue
            if not (r1.get("main_html") or "").strip(): S["no_main_html"] += 1; recs.append({"stem": stem, "idx": idx, "in_md5": md5(raw), "skip": "no_main_html"}); continue
            raws.append(raw); keys.append({"stem": stem, "idx": idx, "in_md5": md5(raw)})
    rendered = pool.map(render, raws, chunksize=16)
    prompts, ninps, pkeys = [], [], []
    for k, txt in zip(keys, rendered):
        if not txt or len(txt) < 20: S["render_fail_or_empty"] += 1; recs.append(dict(k, skip="render_fail_or_empty")); continue
        ninp = E.number_lines(txt); p = prompt_of(ninp)
        if len(tok.encode(p)) > MAX_PROMPT: S["too_long"] += 1; recs.append(dict(k, skip="too_long")); continue
        prompts.append(p); ninps.append(ninp); pkeys.append(k)
    outs = llm.generate(prompts, sp, use_tqdm=False) if prompts else []
    dst = os.path.join(OUT, stem + "_processed.jsonl.gz"); tmp = dst + ".tmp%d" % os.getpid(); n_out = 0
    rdst = os.path.join(RAWD, stem + "_raw.jsonl.gz"); rtmp = rdst + ".tmp%d" % os.getpid()
    with gzip.open(tmp, "wt", encoding="utf-8") as g:
        for ninp, o, k in zip(ninps, outs, pkeys):
            gen = o.outputs[0].text
            rec = dict(k, gen=gen, finish=o.outputs[0].finish_reason, n_gen_tok=len(o.outputs[0].token_ids))
            if SAVE_NINP: rec["ninp"] = ninp
            recs.append(rec)
            try: rec["extracted"] = extracted_of(ninp, gen)
            except Exception as e: S["extract_fail"] += 1; rec["extracted"] = None; rec["extract_err"] = repr(e)[:300]
            rec["tag"] = None; rec["final"] = None; rec["in_full"] = False
            try: tag, text = X.body_from_prediction_dfirst(ninp, o.outputs[0].text)
            except Exception as e: S["parse_fail"] += 1; rec["parse_err"] = repr(e)[:300]; continue
            rec["tag"] = tag; rec["final"] = text
            S["tag" + str(tag)] += 1
            if tag == "<delete>" or not (text or "").strip(): continue
            g.write(json.dumps({"text": text, "e2e_tag": tag}, ensure_ascii=False) + "\n"); n_out += 1; rec["in_full"] = True
    recs.sort(key=lambda r: r["idx"])
    with gzip.open(rtmp, "wt", encoding="utf-8") as rg:
        for r in recs: rg.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(rtmp, rdst)
    os.replace(tmp, dst); S["pages"] += len(raws); S["written"] += n_out; S["raw_rows"] += len(recs)
    print(f"rank {rank} {stem}: pages {len(raws)} kept {n_out} in {time.time()-t0:.0f}s | {dict(S)}", flush=True)
pool.close(); pool.join()
print(f"rank {rank} DONE {dict(S)}", flush=True)
