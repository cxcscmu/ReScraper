"""Pool inference of the two-stage refiner (text in, text out): input = Dripper's extracted text of every pool page
(<pool>/input/<stem>.jsonl.gz, rows {"text"} in step-2 order), output = <pool>/<INFER_MODEL_TAG>/<stem>.jsonl.gz with
one {"text"} row per input row ("" when the model answers <delete>). Each page is split into INFER_CHUNK_CHARS-character
chunks (100,000 in our run, i.e. effectively whole pages), each chunk is prompted with the system prompt in
INFER_SYSTEM_PROMPT_FILE (prompts/two_stage_refiner_system.txt), the leading decision tag is stripped, and chunk
outputs are joined with " ". Oversized prompts are tail-truncated to the 40,960 - 4,096 token budget; the output
budget per request is min(40960 - prompt, prompt + 256). Decoding for the paper run: INFER_TEMPERATURE=0.0,
INFER_TOP_P=1.0 (the script defaults, 1.0 / 0.9, are kept for backward compatibility of older runs).
Resumable per shard (atomic writes).

Usage:
  python infer_t2t.py <rank> <input_dir> <world_size>
"""
from vllm import LLM, SamplingParams
from pathlib import Path
from tqdm import tqdm
import datasets
import gzip
import json
import os
import re
import sys

MODEL_PATH = os.environ["INFER_MODEL_PATH"]
MODEL_TAG = os.environ.get("INFER_MODEL_TAG", "best")

CHUNK_CHARS = int(os.environ.get("INFER_CHUNK_CHARS", "50000"))   # 50k = legacy; the 40,960-token window fits ~120k chars
MAX_NEW_TOKENS = 4096
MODEL_MAX_LEN = 40960
# Pre-truncated in make_conversation. SamplingParams.truncate_prompt_tokens is
# silently ignored by LLM.generate() (it only applies in the OpenAI server
# entrypoints), so we slice the token IDs ourselves before submission.
MAX_PROMPT_TOKENS = MODEL_MAX_LEN - MAX_NEW_TOKENS  # 36864

PREFIX_RE = re.compile(r"^\s*\[(?:extract\+refine|extract)\]\s*\n", re.IGNORECASE)
# The model emits a decision tag before the body. <delete> means the page is discarded,
# so that chunk contributes nothing; the other tags are stripped. PREFIX_RE handles an
# older bracketed prefix format.
TAG_RE = re.compile(r"^\s*<(extract|refine|delete|rewrite)>\s*", re.IGNORECASE)

# Defaults are temperature 1.0 / top_p 0.9; the two-stage refiner pool run used greedy decoding
# (INFER_TEMPERATURE=0.0 INFER_TOP_P=1.0, set by infer_t2t.sbatch).
TEMPERATURE = float(os.environ.get("INFER_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("INFER_TOP_P", "0.9"))
SYS_PROMPT = open(os.environ["INFER_SYSTEM_PROMPT_FILE"], encoding="utf-8").read().strip() if os.environ.get("INFER_SYSTEM_PROMPT_FILE") else None


def budget(plen):
    """Output can never exceed the input it is extracted from, so cap each request
    at its own prompt length plus slack instead of a flat MAX_NEW_TOKENS. vLLM
    stops at EOS anyway, so a higher ceiling costs nothing on normal pages - it
    only stops cutting off the long clean ones."""
    return max(64, min(MODEL_MAX_LEN - plen, plen + 256))


def make_conversation(example, tokenizer, prompt_column="text"):
    text = example[prompt_column]
    msgs = [{"role": "user", "content": text}]
    if SYS_PROMPT:
        msgs = [{"role": "system", "content": SYS_PROMPT}] + msgs
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    token_ids = tokenizer.encode(prompt)
    # If a chunk's templated prompt exceeds the model's input budget, keep the
    # last MAX_PROMPT_TOKENS so the assistant header at the end is preserved.
    # (Truncates the head of the user content; chat scaffolding at start is
    # lost on truncation, but the model still generates a paraphrase.)
    if len(token_ids) > MAX_PROMPT_TOKENS:
        token_ids = token_ids[-MAX_PROMPT_TOKENS:]
    return {"prompt_token_ids": token_ids, "n_prompt": len(token_ids)}


def transform(llm, prompts):
    sps = [SamplingParams(temperature=TEMPERATURE, top_p=TOP_P,
                          max_tokens=budget(len(p["prompt_token_ids"])))
           for p in prompts]
    outputs = llm.generate(prompts, sps, use_tqdm=False)
    out_texts = []
    for o in outputs:
        text = o.outputs[0].text
        m = TAG_RE.match(text)
        if m:
            if m.group(1).lower() == "delete":
                out_texts.append("")      # page judged valueless - drop it
                continue
            text = text[m.end():]
        clean = PREFIX_RE.sub("", text, count=1).strip()
        out_texts.append(clean)
    return out_texts


def vllm_inference(llm, dataset):
    batch_size = 8192
    out = []
    for i in range(0, len(dataset), batch_size):
        token_ids_batch = dataset[i : i + batch_size]["prompt_token_ids"]
        prompts = [{"prompt_token_ids": ids} for ids in token_ids_batch]
        out.extend(transform(llm, prompts))
    return out


def load_jsonl(path: Path):
    if str(path).endswith(".gz"):
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = path.open("r", encoding="utf-8")
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def open_for_write(path: Path, gzipped=None):
    # gzipped MUST reflect the FINAL destination, not the path handed in.
    # Atomic writes go to a ".tmp<pid>" name that does not end in ".gz", so
    # inferring from the temp name writes PLAIN text into a file that is then
    # renamed to *.jsonl.gz.
    if gzipped is None:
        gzipped = str(path).endswith(".gz")
    if gzipped:
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def paths_from_dir(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.jsonl.gz")) + sorted(input_dir.rglob("*.jsonl"))


def stitch(chunks: list[str]) -> str:
    """Concat chunk paraphrases for one source doc."""
    joined = " ".join(c for c in chunks if c)
    if joined.startswith("---"):
        joined = joined[3:]
    if joined.endswith("---"):
        joined = joined[:-3]
    return joined.strip()


def main():
    rank = int(sys.argv[1])
    input_dir = Path(sys.argv[2])
    world_size = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    print(f"rank={rank} world_size={world_size} input_dir={input_dir}")

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=40960,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
    )
    tokenizer = llm.get_tokenizer()

    read_paths = paths_from_dir(input_dir)  # forward global order
    # Opt-in tail-first consumption. Two arrays over the same pool that both
    # walk forward converge on the same frontier and recompute each other's
    # in-flight shards; running the second one with INFER_REVERSE=1 makes them
    # meet in the middle instead. Default off -> existing arrays unaffected.
    if os.environ.get("INFER_REVERSE", "0") == "1":
        read_paths = read_paths[::-1]
        print(f"rank={rank} INFER_REVERSE=1 (consuming shard list from the tail)")
    read_paths = [p for i, p in enumerate(read_paths) if i % world_size == rank]
    limit = int(os.environ.get("LIMIT_FILES", "0"))
    if limit > 0:
        read_paths = read_paths[:limit]
        print(f"rank={rank} LIMIT_FILES={limit} → {len(read_paths)} files")
    print(f"rank={rank} assigned {len(read_paths)} files (forward order)")

    for read_file_path in tqdm(read_paths, desc=f"rank={rank}"):
        rel = read_file_path.relative_to(input_dir)
        write_file_path = input_dir.parent / MODEL_TAG / rel
        if write_file_path.exists():
            print(f"skip (exists): {write_file_path}")
            continue
        print(f"processing: {read_file_path}")

        # Per source row: split into CHUNK_CHARS chunks, track tid (row id).
        chunk_rows = []
        chunk_tids = []
        n_orig = 0
        for tid, item in enumerate(load_jsonl(read_file_path)):
            text = item["text"]
            if not text:
                # Empty input → still emit one empty chunk slot so output
                # alignment is preserved.
                chunk_rows.append({"text": ""})
                chunk_tids.append(tid)
            else:
                for j in range(0, len(text), CHUNK_CHARS):
                    chunk_rows.append({"text": text[j : j + CHUNK_CHARS]})
                    chunk_tids.append(tid)
            n_orig = tid + 1
        if n_orig == 0:
            print(f"empty file, writing empty output: {write_file_path}")
            write_file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = write_file_path.with_suffix(write_file_path.suffix + f".tmp{os.getpid()}")
            with open_for_write(tmp_path, str(write_file_path).endswith(".gz")) as f:
                pass
            os.replace(tmp_path, write_file_path)
            continue

        # Templating + pre-truncation in make_conversation. No token filter:
        # empty-text chunks are skipped (no work to do); oversized chunks are
        # tail-truncated to MAX_PROMPT_TOKENS so vLLM never sees > model_max.
        rows_with_idx = [{"text": r["text"], "_chunk_idx": k} for k, r in enumerate(chunk_rows)]
        ds = datasets.Dataset.from_list(rows_with_idx).map(
            lambda x: make_conversation(x, tokenizer) if x["text"] else {"prompt_token_ids": []}
        )
        n_total = len(ds)
        ds_kept = ds.filter(lambda x: x["text"])
        n_kept = len(ds_kept)
        print(f"  rows={n_orig} chunks={n_total} kept={n_kept}")

        # Run inference on kept chunks; rebuild full chunk list with empties.
        kept_chunk_idx = ds_kept["_chunk_idx"]
        gen_texts = vllm_inference(llm, ds_kept) if n_kept > 0 else []
        gen_by_chunk = dict(zip(kept_chunk_idx, gen_texts))

        # Group chunks by tid in order, then stitch.
        per_row: dict[int, list[str]] = {}
        for k, tid in enumerate(chunk_tids):
            per_row.setdefault(tid, []).append(gen_by_chunk.get(k, ""))

        write_file_path.parent.mkdir(parents=True, exist_ok=True)
        empty_count = 0
        tmp_path = write_file_path.with_suffix(write_file_path.suffix + f".tmp{os.getpid()}")
        with open_for_write(tmp_path, str(write_file_path).endswith(".gz")) as f:
            for tid in range(n_orig):
                stitched = stitch(per_row.get(tid, []))
                if not stitched:
                    empty_count += 1
                f.write(json.dumps({"text": stitched}) + "\n")
        os.replace(tmp_path, write_file_path)
        print(f"  wrote {n_orig} rows, {empty_count} empty after stitch")


if __name__ == "__main__":
    main()
