"""gpt-oss-120b output judge (Appendix "Output judge"): is the TEXT a pipeline kept from a page worth keeping in a
pretraining corpus as it stands (keep/remove + informative-content value 0-3)? The judge sees only the kept text, never the
source page or any classifier score. Used for the page-quality check of Figure 6 (appendix table of mean values by the
Figure 6 groups) and for Figure 1(b), where a junk page kept only as text the judge rates worth keeping counts as removed.
Prompt: prompts/judge_output.txt (system message); user message = "TEXT:\\n<<<\\n<text>\\n>>>".
vLLM settings as keep_judge.py (same code path): LLM.chat, temperature 0, reasoning_effort low, max_tokens 2048,
chat-template date pinned (JUDGE_DATE), max_model_len 32768, MAXCH 24000 with a token guard, last-valid-JSON parsing.
usage: output_judge.py build <heldout_pipeline work dir of the release run> <out_in.jsonl>
         (one row per non-empty output of RefinedWeb-rule, FineWeb-rule, ProX-C, UltraX and ReScraper: gid, system, text)
       output_judge.py <in.jsonl> <out.jsonl>     env: JUDGE_MODEL TP GPU_UTIL MAXCH MAX_TOKENS LIMIT JUDGE_DATE"""
import json, os, sys, time, glob
MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
TP = int(os.environ.get("TP", "1")); GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.94"))
MAXCH = int(os.environ.get("MAXCH", "24000")); MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048")); MAX_LEN = 32768
LIMIT = int(os.environ.get("LIMIT", "0")); JUDGE_DATE = os.environ.get("JUDGE_DATE", "2026-09-23")
RUB = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prompts", "judge_output.txt"), encoding="utf-8").read().strip()
TRUNC = "\n[... truncated for length]"; DEC = json.JSONDecoder()
def clip(s, n): return s if len(s) <= n else s[:n] + TRUNC
def msgs(text, n): return [{"role": "system", "content": RUB}, {"role": "user", "content": "TEXT:\n<<<\n" + clip(text, n) + "\n>>>"}]
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
def parse(t):
    o = last_json(t, ["value", "verdict"])
    if o is None: return None
    try: v = int(o["value"])
    except (TypeError, ValueError): return None
    vd = str(o["verdict"]).strip().lower()
    if v not in (0, 1, 2, 3) or vd not in ("keep", "remove"): return None
    return {"value": v, "verdict": vd, "reason": str(o.get("reason", ""))[:300]}
def pinned_template(model_path):
    c = glob.glob(os.path.join(model_path, "chat_template.jinja"))
    if not c: return None
    t = open(c[0], encoding="utf-8").read(); assert t.count('strftime_now("%Y-%m-%d")') == 1
    return t.replace('strftime_now("%Y-%m-%d")', '"%s"' % JUDGE_DATE)

SYSTEMS = (("RefinedWeb-rule", "sys_refinedweb_rule.jsonl"), ("FineWeb-rule", "sys_fineweb_rule.jsonl"),
           ("ProX-C", "sys_proxc.jsonl"), ("UltraX", "sys_ultrax.jsonl"), ("ReScraper", "ours.jsonl"))
def build(work_dir, dst):
    """Every non-empty output of the Figure 6 pipelines on the held-out pages, in the order of SYSTEMS."""
    n = 0
    with open(dst, "w", encoding="utf-8") as f:
        for name, fn in SYSTEMS:
            for r in map(json.loads, open(os.path.join(work_dir, fn), encoding="utf-8")):
                t = r.get("text")
                if t and t != "None" and t.strip():
                    f.write(json.dumps({"gid": int(r["gid"]), "system": name, "text": t}, ensure_ascii=False) + "\n"); n += 1
    print(n, "texts ->", dst)
def main():
    if sys.argv[1] == "build": build(sys.argv[2], sys.argv[3]); return
    src, dst = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    if LIMIT: rows = rows[:LIMIT]
    from vllm import LLM, SamplingParams
    from huggingface_hub import snapshot_download
    llm = LLM(model=MODEL, tensor_parallel_size=TP, dtype="auto", max_model_len=MAX_LEN, gpu_memory_utilization=GPU_UTIL)
    tok = llm.get_tokenizer(); sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    tmpl = pinned_template(snapshot_download(MODEL, local_files_only=True))
    ms, used = [], []
    for r in rows:
        n = MAXCH
        while True:
            m = msgs(r["text"], n)
            if len((m[0]["content"] + m[1]["content"]).encode("utf-8")) + 512 + MAX_TOKENS <= MAX_LEN: break
            if len(tok.encode(m[0]["content"] + m[1]["content"])) + 512 + MAX_TOKENS <= MAX_LEN or n < 500: break
            n = int(n * 0.8)
        ms.append(m); used.append(n)
    t0 = time.time()
    outs = llm.chat(ms, sp, use_tqdm=False, chat_template=tmpl, chat_template_kwargs={"reasoning_effort": "low"})
    ok = nlen = 0
    with open(dst + ".tmp", "w", encoding="utf-8") as f:
        for r, o, n in zip(rows, outs, used):
            txt = o.outputs[0].text; fr = o.outputs[0].finish_reason; p = parse(txt); ok += p is not None; nlen += fr == "length"
            f.write(json.dumps({"gid": r["gid"], "system": r["system"], "parsed": p, "raw": txt, "finish_reason": fr, "maxch_used": n,
                                "truncated": len(r["text"]) > n, "model": MODEL, "gpu": os.environ.get("GPU_NAME", "?"),
                                "job": os.environ.get("SLURM_JOB_ID", "?")}, ensure_ascii=False) + "\n")
    os.replace(dst + ".tmp", dst)
    print("OUTPUTJUDGE %d req, parsed %d (%.2f%%), finish=length %d, %.1f min" % (len(rows), ok, 100.0 * ok / max(len(rows), 1), nlen, (time.time() - t0) / 60), flush=True)
    if ok < 0.99 * len(rows): sys.exit(3)
    print("OUTPUTJUDGE_OK")
if __name__ == "__main__": main()
