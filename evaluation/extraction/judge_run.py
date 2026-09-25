"""gpt-oss-120b extraction judge (Figure 8, right): pointwise, no reference extraction, one model load per job.
vLLM LLM.chat, temperature 0, chat_template_kwargs reasoning_effort=low, max_model_len 32768, MAXCH=24000 character
truncation per text, max_tokens 2048, last-valid-JSON parsing (preferring the text after the last "assistantfinal"
channel marker), raw output + finish_reason kept. The chat template's "Current date" is pinned (JUDGE_DATE, default
2026-09-23, the date of the paper's runs) so reruns see byte-identical prompts; truncated texts end with
"[... truncated for length]" (the prompt tells the judge so); a token guard shrinks MAXCH for a request whose prompt would
not fit; resumable work units; gid sharding.
Prompt: prompts/judge_extraction.txt (system message); user message = "PAGE:\\n<<<\\n<rendered page>\\n>>>\\n\\nEXTRACTION:\\n<<<\\n
<extraction>\\n>>>" (empty extraction -> "(empty: the extractor returned no text)").
Tasks (JUDGE_TASKS, run in this order):
  sanity  50 pages x 5 synthetic controls (empty / full rendered page / first 40% of Dripper lines / Dripper text
          twice / words shuffled within each line); run first; the job stops (exit 3) if < 99% of them parse.
  extract every page x 5 systems (student, dripper, resiliparse, trafilatura, justext); per page the systems are
          submitted in a random order (random.Random(gid).shuffle) and that position is recorded.
usage: judge_run.py <ext_texts.jsonl> <out_dir>
env: JUDGE_MODEL TP GPU_UTIL MAXCH MAX_TOKENS UNIT JUDGE_TASKS SHARD_ID SHARD_N LIMIT GPU_NAME JOB_ID EXT_PROMPT JUDGE_DATE
     EXT_SYSTEMS
"""
import json, os, sys, time, random, re
PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prompts")
MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
TP = int(os.environ.get("TP", "1")); GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.9"))
MAXCH = int(os.environ.get("MAXCH", "24000")); MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048")); MAX_LEN = 32768
UNIT = int(os.environ.get("UNIT", "1000")); TASKS = re.split(r"[,+]", os.environ.get("JUDGE_TASKS", "sanity,extract"))   # "+" also accepted (sbatch --export splits on ",")
SID = int(os.environ.get("SHARD_ID", os.environ.get("SLURM_ARRAY_TASK_ID", "0"))); SN = int(os.environ.get("SHARD_N", "1"))
LIMIT = int(os.environ.get("LIMIT", "0"))
EXT_PROMPT = os.environ.get("EXT_PROMPT", os.path.join(PROMPTS, "judge_extraction.txt"))
RUB_EXT = open(EXT_PROMPT, encoding="utf-8").read().strip()
EXT_SYSTEMS = ["student", "dripper", "resiliparse", "trafilatura", "justext"]
# optional subset (rerun only what changed, e.g. a new student output): EXT_SYSTEMS=student
# (the per-page submission position is still taken from the full 5-system shuffle, so it matches a full run)
EXT_ONLY = [x for x in re.split(r"[,+]", os.environ.get("EXT_SYSTEMS", "")) if x]
EXT_KEYS = ["main_content_recall", "boilerplate_precision", "integrity"]
TRUNC = "\n[... truncated for length]"
# The gpt-oss chat template writes "Current date: <today>" into the system message (strftime_now). Pin it so reruns see
# byte-identical prompts: the template is the model's own, with that call replaced by JUDGE_DATE.
JUDGE_DATE = os.environ.get("JUDGE_DATE", "2026-09-23")
def pinned_template(model_path):
    import glob
    c = glob.glob(os.path.join(model_path, "chat_template.jinja"))
    if not c: return None
    t = open(c[0], encoding="utf-8").read(); assert t.count('strftime_now("%Y-%m-%d")') == 1
    return t.replace('strftime_now("%Y-%m-%d")', '"%s"' % JUDGE_DATE)
DEC = json.JSONDecoder()

def clip(s, n): return s if len(s) <= n else s[:n] + TRUNC
def ext_msgs(page, text, n):
    body = ("PAGE:\n<<<\n" + clip(page, n) + "\n>>>\n\nEXTRACTION:\n<<<\n" +
            (clip(text, n) if text.strip() else "(empty: the extractor returned no text)") + "\n>>>")
    return [{"role": "system", "content": RUB_EXT}, {"role": "user", "content": body}]

def last_json(text, need):
    t = text or ""
    segs = ([t.rsplit("assistantfinal", 1)[1]] if "assistantfinal" in t else []) + [t]
    for seg in segs:
        i = len(seg)
        while True:
            i = seg.rfind("{", 0, i)
            if i < 0: break
            try: obj, _ = DEC.raw_decode(seg, i)
            except ValueError: continue
            if isinstance(obj, dict) and all(k in obj for k in need): return obj
    return None
def parse_ext(t):
    o = last_json(t, EXT_KEYS)
    if o is None: return None
    try: sc = {k: max(0, min(2, int(o[k]))) for k in EXT_KEYS}
    except Exception: return None
    sc["rationale"] = str(o.get("rationale", "")); return sc

def mine(g): return g % SN == SID

def sanity_controls(texts):
    """50 pages with a clearly identifiable main content: Dripper >= 5 lines and >= 100 words, and the rendered page
    >= 1.3x the Dripper text in characters (so it carries real boilerplate). Fixed: the first 50 such gids."""
    sel = [t for t in texts if len([l for l in t["dripper"].split("\n") if l.strip()]) >= 5 and len(t["dripper"].split()) >= 100
           and len(t["rendered"]) >= 1.3 * len(t["dripper"])][:50]
    out = []
    for t in sel:
        dl = [l for l in t["dripper"].split("\n") if l.strip()]; rng = random.Random(t["gid"])
        def shuf(l):
            w = l.split(); rng.shuffle(w); return " ".join(w)
        ctl = {"ctl_empty": "", "ctl_full": t["rendered"], "ctl_first40": "\n".join(dl[:max(1, int(0.4 * len(dl)))]),
               "ctl_twice": t["dripper"] + "\n" + t["dripper"], "ctl_shuffled": "\n".join(shuf(l) for l in dl)}
        for s, x in ctl.items(): out.append({"task": "sanity", "gid": t["gid"], "system": s, "page": t["rendered"], "text": x})
    return out

def build(ext_path):
    reqs = {"sanity": [], "extract": []}
    texts = [json.loads(l) for l in open(ext_path, encoding="utf-8")]
    if LIMIT: texts = texts[:LIMIT]
    if "sanity" in TASKS and SID == 0: reqs["sanity"] = sanity_controls(texts)
    for t in texts:
        if not mine(t["gid"]): continue
        order = EXT_SYSTEMS[:]; random.Random(t["gid"]).shuffle(order)
        for pos, s in enumerate(order):
            if EXT_ONLY and s not in EXT_ONLY: continue
            reqs["extract"].append({"task": "extract", "gid": t["gid"], "system": s, "pos": pos, "page": t["rendered"], "text": t[s] or ""})
    return reqs

def fit(tok, r):
    """messages + the MAXCH actually used; shrink by 20% steps until prompt + MAX_TOKENS fits the context"""
    n = MAXCH
    while True:
        m = ext_msgs(r["page"], r["text"], n)
        nb = len((m[0]["content"] + m[1]["content"]).encode("utf-8"))
        if nb + 512 + MAX_TOKENS <= MAX_LEN: return m, n          # a BPE token covers >= 1 byte
        ntok = len(tok.encode(m[0]["content"] + m[1]["content"])) + 512
        if ntok + MAX_TOKENS <= MAX_LEN or n < 500: return m, n
        n = int(n * 0.8)

TEMPLATE = None
def run_unit(llm, sp, tok, rs, path):
    msgs, used = [], []
    for r in rs:
        m, n = fit(tok, r); msgs.append(m); used.append(n)
    t0 = time.time()
    outs = llm.chat(msgs, sp, use_tqdm=False, chat_template=TEMPLATE, chat_template_kwargs={"reasoning_effort": "low"})
    ok = nlen = 0
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r, o, n in zip(rs, outs, used):
            txt = o.outputs[0].text; fr = o.outputs[0].finish_reason; nlen += fr == "length"
            sc = parse_ext(txt); ok += sc is not None
            rec = {k: r[k] for k in ("task", "gid", "system", "pos") if k in r}
            trunc = any(len(x) > n for x in (r["page"], r["text"]))
            rec.update({"parsed": sc, "raw": txt, "finish_reason": fr, "n_out_tokens": len(o.outputs[0].token_ids),
                        "n_in_tokens": len(o.prompt_token_ids or []), "maxch_used": n, "truncated": trunc, "model": MODEL,
                        "gpu": os.environ.get("GPU_NAME", "?"), "tp": TP, "job": os.environ.get("JOB_ID", "?")})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    dt = time.time() - t0
    print("[%s] %s: %d req, parsed %d, finish=length %d, %.1f min (%.2f req/s)" % (time.strftime("%H:%M:%S"), os.path.basename(path),
          len(rs), ok, nlen, dt / 60, len(rs) / max(dt, 1e-9)), flush=True)
    return ok, len(rs)

def main():
    ext_path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out + "/units", exist_ok=True)
    reqs = build(ext_path)
    tag = "s%dof%d" % (SID, SN)
    plan = []
    for task in ("sanity", "extract"):
        if task not in TASKS: continue
        rs = reqs[task]
        for k in range(0, len(rs), UNIT):
            plan.append((task, rs[k:k + UNIT], "%s/units/%s_%s_u%04d.jsonl" % (out, task, tag, k // UNIT)))
    todo = [u for u in plan if not os.path.exists(u[2])]
    print("extract prompt:", EXT_PROMPT, "| EXT_SYSTEMS", EXT_ONLY or "all", flush=True)
    print("model %s tp %d gpu %s | shard %s | requests: %s | units %d, todo %d" % (MODEL, TP, os.environ.get("GPU_NAME"), tag,
          {k: len(v) for k, v in reqs.items()}, len(plan), len(todo)), flush=True)
    if not todo:
        print("JUDGE_RUN_DONE (nothing to do)", flush=True); return
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=TP, dtype="auto", max_model_len=MAX_LEN, gpu_memory_utilization=GPU_UTIL)
    print("model load %.1f min" % ((time.time() - t0) / 60), flush=True)
    tok = llm.get_tokenizer(); sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    global TEMPLATE
    try:
        mp = llm.llm_engine.model_config.model
    except Exception:
        mp = MODEL
    if not os.path.isdir(mp):
        from huggingface_hub import snapshot_download
        mp = snapshot_download(MODEL, local_files_only=True)
    TEMPLATE = pinned_template(mp)
    print("chat template: %s" % ("model template, date pinned to " + JUDGE_DATE if TEMPLATE else "model default (date NOT pinned)"), flush=True)
    for task, rs, path in todo:
        ok, n = run_unit(llm, sp, tok, rs, path)
        if task == "sanity" and ok < 0.99 * n:
            print("SANITY_PARSE_FAIL %d / %d" % (ok, n), flush=True); sys.exit(3)
    print("JUDGE_RUN_DONE total %.1f min" % ((time.time() - t0) / 60), flush=True)

if __name__ == "__main__":
    main()
