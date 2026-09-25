"""Stage 1a (GPU): run the released student on every page of the page table (greedy run).

Decoding (the held-out evaluation settings of the SFT runs, decision-first staged reader):
  checkpoint $STAGE2_CKPT, vLLM LLM(dtype=bfloat16, gpu_memory_utilization=0.90, max_model_len=32768)
  messages = [system: prompts/student_system_stage2.txt .strip(), user: page["input"]] -> apply_chat_template(
             add_generation_prompt=True, enable_thinking=False)
  pages with plen + 256 > 32768 are not sent -> status "prompt_too_long"
  SamplingParams(temperature=0.0, max_tokens=min(8192, 32768 - plen))
Pages are generated in chunks of CHUNK (default 1000) and each chunk is written as soon as it is done, so a requeued job
resumes.
Output: <work>/ours_raw.jsonl  {gid, status, raw, plen, tlen (teacher output tokens), gen_tokens, finish_reason}
The release-decoding records (T=1.0, top_p=1.0, max_tokens=3072) come from stages/s1_rel_extract.py; they take plen/tlen
from this file.
usage: s1_ours_infer.py <page_table>
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from common import *
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

CKPT = os.environ.get("STAGE2_CKPT") or sys.exit("set STAGE2_CKPT (the released ReScraper checkpoint)")
PT = sys.argv[1]; W = workdir(PT); OUT = W + "/ours_raw.jsonl"; PARTS = W + "/ours_raw_parts"
os.makedirs(PARTS, exist_ok=True)
CHUNK = int(os.environ.get("OURS_CHUNK", "1000"))
MAXLEN, MIN_GEN = 32768, 256
pages = load_pages(PT, ["gid", "input", "output"])
SYS = open(SYS_PROMPT_FILE, encoding="utf-8").read().strip()
tok = AutoTokenizer.from_pretrained(CKPT)
chunks = [pages[i:i + CHUNK] for i in range(0, len(pages), CHUNK)]
todo = [c for c in range(len(chunks)) if not os.path.exists("%s/part%03d.jsonl" % (PARTS, c))]
print("pages %d chunks %d todo %s" % (len(pages), len(chunks), todo), flush=True)
llm = LLM(model=CKPT, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=MAXLEN) if todo else None
t0 = time.time()
for c in todo:
    recs, prompts, sps, idx = [], [], [], []
    for p in chunks[c]:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": p["input"]}]
        pr = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        plen = len(tok(pr, add_special_tokens=False).input_ids)
        tlen = len(tok(p["output"], add_special_tokens=False).input_ids) if p.get("output") else None
        r = {"gid": p["gid"], "status": "ok", "raw": None, "plen": plen, "tlen": tlen, "gen_tokens": None, "finish_reason": None}
        if plen + MIN_GEN > MAXLEN:
            r["status"] = "prompt_too_long"
        else:
            idx.append(len(recs)); prompts.append(pr)
            sps.append(SamplingParams(temperature=0.0, max_tokens=min(8192, MAXLEN - plen)))
        recs.append(r)
    outs = llm.generate(prompts, sps, use_tqdm=False) if prompts else []
    for j, o in zip(idx, outs):
        recs[j].update(raw=o.outputs[0].text, gen_tokens=len(o.outputs[0].token_ids), finish_reason=o.outputs[0].finish_reason)
    write_jsonl_atomic("%s/part%03d.jsonl" % (PARTS, c), recs)
    print("chunk %d/%d done: %d pages (%d generated)  %.0fs" % (c + 1, len(chunks), len(recs), len(prompts), time.time() - t0), flush=True)
allr = []
for c in range(len(chunks)):
    allr += list(read_jsonl("%s/part%03d.jsonl" % (PARTS, c)))
assert [r["gid"] for r in allr] == list(range(len(pages)))
write_jsonl_atomic(OUT, allr)
print("status", {s: sum(r["status"] == s for r in allr) for s in {r["status"] for r in allr}})
print("OURS_INFER_OK", OUT, flush=True)
