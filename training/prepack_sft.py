"""Offline tokenize + pack an SFT jsonl once (CPU node), so 32-64 training ranks do not each
re-tokenize 1M samples and bin-pack them in Python lists (which ran the host out of memory).
Reproduces sft_train.py exactly: same train/val split (seed 42), same preprocess/build_chat_ids
with the system prompt, same length filter; packing is best-fit-decreasing (a valid packing, the
in-script FFD differs only in which samples share a bin). Output: DatasetDict(train=packed int32
input_ids/labels/position_ids, validation=tokenized) + val_raw.json, loaded by sft_train.py --prepacked_dir."""
import os, sys, json, argparse, bisect, collections, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_train as S
from datasets import load_dataset, DatasetDict, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--data_path", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--model_path", default="Qwen/Qwen3-0.6B"); ap.add_argument("--system_prompt_file", default=None)
ap.add_argument("--max_seq_len", type=int, default=32768); ap.add_argument("--val_size", type=int, default=1000)
ap.add_argument("--max_samples", type=int, default=0); ap.add_argument("--num_proc", type=int, default=48)
a = ap.parse_args(); t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "right"
if a.system_prompt_file:
    S.SYSTEM_PROMPT = open(a.system_prompt_file, encoding="utf-8").read().strip(); print("system prompt chars", len(S.SYSTEM_PROMPT))
data = load_dataset("json", data_files=a.data_path, split="train")
if a.max_samples > 0: data = data.select(range(min(a.max_samples, len(data))))
val_size = min(a.val_size, len(data) // 5)
split = data.train_test_split(test_size=val_size, seed=42)
ds = DatasetDict({"train": split["train"], "validation": split["test"]})
MAX = a.max_seq_len; PREF = MAX * 3
def preprocess(ex):
    hi, to = ex["input"], ex["output"]
    if len(hi) + len(to) > PREF: return {"input_ids": [], "attention_mask": [], "labels": []}
    full = S.build_chat_ids(tok, hi, to); prompt = S.build_chat_ids(tok, hi, "", add_generation_prompt=True)
    pl = min(len(prompt), len(full)); labels = ([-100] * pl + full[pl:])[: len(full)]
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}
ds["validation"] = ds["validation"].map(preprocess, num_proc=min(a.num_proc, 8), desc="Tokenizing validation")
ds["train"] = ds["train"].map(preprocess, remove_columns=["input", "output"], num_proc=a.num_proc, desc="Tokenizing train")
before = {k: len(ds[k]) for k in ds}
ds = ds.filter(lambda x: 0 < len(x["input_ids"]) <= MAX, num_proc=a.num_proc)
for k in ds: print(k, before[k], "->", len(ds[k]))
val_raw = [{"input": ex["input"], "output": ex["output"]} for ex in ds["validation"]]
ds["validation"] = ds["validation"].remove_columns(["input", "output"])
print("tokenized in %.0fs" % (time.time() - t0), flush=True)
# lengths without materialising the ids in python
L = ds["train"].map(lambda ex: {"n": len(ex["input_ids"])}, remove_columns=ds["train"].column_names, num_proc=a.num_proc, desc="lengths")["n"]
order = sorted(range(len(L)), key=lambda i: -L[i]); bins = []; nonempty = []; at = collections.defaultdict(list)
for i in order:
    s = L[i]; j = bisect.bisect_left(nonempty, s)
    if j < len(nonempty):
        c = nonempty[j]; b = at[c].pop()
        if not at[c]: del nonempty[j]
    else:
        b = len(bins); bins.append([]); c = MAX
    bins[b].append(i); nc = c - s
    if nc > 0:
        if not at[nc]: bisect.insort(nonempty, nc)
        at[nc].append(b)
tot = sum(L); print("packing: %d samples -> %d bins (ratio %.2fx, avg fill %.1f%%) in %.0fs" % (len(L), len(bins), len(L) / max(1, len(bins)), 100.0 * tot / max(1, len(bins)) / MAX, time.time() - t0), flush=True)
flat = [i for b in bins for i in b]; sizes = [len(b) for b in bins]
sub = ds["train"].select(flat)
def gen():
    it = iter(sub)
    for nb in sizes:
        ids, labs, pos = [], [], []
        for _ in range(nb):
            ex = next(it); n = len(ex["input_ids"]); ids.extend(ex["input_ids"]); labs.extend(ex["labels"]); pos.extend(range(n))
        yield {"input_ids": ids, "labels": labs, "position_ids": pos}
feats = Features({"input_ids": Sequence(Value("int32")), "labels": Sequence(Value("int32")), "position_ids": Sequence(Value("int32"))})
packed = Dataset.from_generator(gen, features=feats, cache_dir=os.environ.get("HF_DATASETS_CACHE"))
os.makedirs(a.out_dir, exist_ok=True)
DatasetDict({"train": packed, "validation": ds["validation"]}).save_to_disk(a.out_dir)
json.dump(val_raw, open(os.path.join(a.out_dir, "val_raw.json"), "w", encoding="utf-8"), ensure_ascii=False)
json.dump({"data_path": a.data_path, "system_prompt_file": a.system_prompt_file, "samples": len(L), "bins": len(bins), "tokens": tot, "val": len(val_raw), "max_seq_len": MAX}, open(os.path.join(a.out_dir, "stats.json"), "w"), indent=1)
print("saved", a.out_dir, "bins", len(packed), "tokens", tot, "val", len(val_raw), "total %.0fs" % (time.time() - t0))
