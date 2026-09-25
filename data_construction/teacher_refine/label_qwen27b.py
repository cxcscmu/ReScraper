#!/usr/bin/env python
"""Refining teacher: Qwen3.8-27B over Dripper's extracted text of every SFT seed page, under the
FineWeb-optimized refinement prompt published with UltraX (prompts/teacher_refine_qwen27b.txt).

Input is dripper's extracted text (not HTML): the prompt is a strict-subset
contract, so the model may only delete. Output per row is either the refined text
or the exact deletion marker, which the builder maps to the three tags.

Striped by rank and resumable: each rank writes its own shard and skips it if the
shard already exists, so a preempted array task costs only its own shard.
Greedy decoding (T=0), thinking disabled, max_tokens = min(32768 - prompt, text + 256), bf16.
Output rows: {"i", "refined" ("" when deleted), "deleted", "finish"}; a row is deleted when the output is
empty or contains the marker "[Content valueless, deleted]" in its first 120 characters.
The same script labels the held-out pages (T27_SRC / T27_DST point at the held-out page table).
env: WORK_DIR, SFT_DIR, T27_SRC (default $SFT_DIR/seed/seed_pages.jsonl), T27_DST (default $SFT_DIR/teacher_refine),
     TEACHER_MODEL (default Qwen/Qwen3.8-27B)
Usage: label_qwen27b.py <rank> <world> [limit]
"""
import glob, json, os, sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

R = os.environ["WORK_DIR"]; SFT = os.environ.get("SFT_DIR", R + "/sft_data")
SRC = os.environ.get("T27_SRC", SFT + "/seed/seed_pages.jsonl")
TEMP = 0.0; DST = os.environ.get("T27_DST", SFT + "/teacher_refine")
PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prompts", "teacher_refine_qwen27b.txt")
MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen3.8-27B")
MAXLEN = 32768
DEL_MARK = "[Content valueless, deleted]"

rank, world = int(sys.argv[1]), int(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(DST, exist_ok=True)
out_path = f"{DST}/part-{rank:03d}-of-{world:03d}.jsonl"
if os.path.exists(out_path):
    print(f"rank {rank}: {out_path} exists, skipping")
    raise SystemExit(0)

rows = []
for n, line in enumerate(open(SRC, encoding="utf-8")):
    if n % world != rank:
        continue
    d = json.loads(line)
    rows.append({"i": d.get("i", n), "text": d.get("output") or ""})   # sources without "i" are keyed by line index
print(f"rank {rank}/{world}: {len(rows)} rows  model={MODEL}", flush=True)

system = open(PROMPT, encoding="utf-8").read().strip()
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
          max_model_len=MAXLEN, trust_remote_code=True)

prompts, sps, keep = [], [], []
skipped = 0
for r in rows:
    t = r["text"]
    if not t.strip():
        skipped += 1
        continue
    p = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": t}],
        tokenize=False, add_generation_prompt=True,
        # Qwen3.8's template turns thinking on by default at xhigh effort, and the
        # chain of thought comes out in place of the answer ("We need answer user's
        # request...") while eating the whole token budget. This is an extraction
        # task with a strict-subset contract - there is nothing to reason about.
        enable_thinking=False)
    plen = len(tok(p, add_special_tokens=False).input_ids)
    tlen = len(tok(t, add_special_tokens=False).input_ids)
    if plen + 64 > MAXLEN:
        skipped += 1
        continue
    keep.append(r)
    prompts.append(p)
    # a strict subset can never be longer than its source, plus slack for the marker
    sps.append(SamplingParams(temperature=TEMP, top_p=(0.95 if TEMP > 0 else 1.0), 
                              max_tokens=min(MAXLEN - plen, tlen + 256)))
print(f"  sent {len(keep)}, skipped {skipped} (empty text or prompt does not fit)", flush=True)

outs = llm.generate(prompts, sps, use_tqdm=True)
tmp = out_path + ".tmp"
n_del = 0
with open(tmp, "w", encoding="utf-8") as f:
    for r, o in zip(keep, outs):
        t = o.outputs[0].text.strip()
        deleted = (not t) or DEL_MARK.lower() in t.lower()[:120]
        if deleted:
            n_del += 1
        f.write(json.dumps({"i": r["i"], "refined": "" if deleted else t,
                            "deleted": deleted,
                            "finish": o.outputs[0].finish_reason},
                           ensure_ascii=False) + "\n")
os.rename(tmp, out_path)
print(f"rank {rank} done: {len(keep)} rows, deleted {n_del} ({100*n_del/max(1,len(keep)):.2f}%) -> {out_path}")
