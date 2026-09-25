"""Merge sharded prepack parts into one prepack dir that sft_train.py --prepacked_dir accepts.
usage: prepack_merge.py <out_base> <n_shards>   -> writes <out_base>/{train,validation,val_raw.json,stats.json}"""
import os, sys, json, shutil
from datasets import load_from_disk, concatenate_datasets, DatasetDict
base, n = sys.argv[1], int(sys.argv[2])
parts = [os.path.join(base, "parts", "%02d" % i) for i in range(n)]
for p in parts:
    assert os.path.exists(os.path.join(p, "stats.json")), "missing shard " + p
st = [json.load(open(os.path.join(p, "stats.json"))) for p in parts]
trains = [load_from_disk(p)["train"] for p in parts]
val = load_from_disk(parts[0])["validation"]
train = concatenate_datasets(trains)
tmp = base + ".merge_tmp"
if os.path.exists(tmp): shutil.rmtree(tmp)
DatasetDict({"train": train, "validation": val}).save_to_disk(tmp)
shutil.copy(os.path.join(parts[0], "val_raw.json"), os.path.join(tmp, "val_raw.json"))
merged = {"data_path": st[0]["data_path"].replace("/shards/part_00.jsonl", " (sharded x%d)" % n), "system_prompt_file": st[0]["system_prompt_file"],
          "samples": sum(s["samples"] for s in st), "bins": sum(s["bins"] for s in st), "tokens": sum(s["tokens"] for s in st),
          "val": st[0]["val"], "max_seq_len": st[0]["max_seq_len"], "shards": n}
json.dump(merged, open(os.path.join(tmp, "stats.json"), "w"), indent=1)
for name in ("train", "validation", "val_raw.json", "stats.json", "dataset_dict.json"):
    dst = os.path.join(base, name)
    if os.path.exists(dst): shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
    shutil.move(os.path.join(tmp, name), dst)
shutil.rmtree(tmp, ignore_errors=True)
print("MERGED", base, merged)
