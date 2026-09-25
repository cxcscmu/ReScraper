"""Sharded prepack. One array task = one shard.

prepack_sft.py is single-node and its packed-dataset save is a single-process generator, so a 1M
prepack takes 1-3 h. This wrapper streams every N-th row of the jsonl into a shard file and runs the
UNCHANGED prepack_sft.py on it; prepack_merge.py then concatenates the shard outputs. Per-shard
packing is a valid packing of the same samples (bins never mix shards), so the trainer sees the same
tokens, prompt and labels. Validation (1,000 rows) comes from shard 0 only; other shards keep 50.
usage: prepack_shard.py <data.jsonl> <out_base> <system_prompt> <shard_id> <n_shards>
"""
import os, sys, subprocess
data, base, sp, i, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
U = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("REFINER_PY", sys.executable)
part = os.path.join(base, "parts", "%02d" % i); shard = os.path.join(base, "shards", "part_%02d.jsonl" % i)
if os.path.exists(os.path.join(part, "stats.json")):
    print("shard", i, "already done"); sys.exit(0)
os.makedirs(os.path.dirname(shard), exist_ok=True)
k = 0
with open(data, encoding="utf-8") as f, open(shard + ".tmp", "w", encoding="utf-8") as g:
    for j, line in enumerate(f):
        if j % n == i:
            g.write(line); k += 1
os.replace(shard + ".tmp", shard); print("shard", i, "rows", k, flush=True)
cmd = [PY, U + "/prepack_sft.py", "--data_path", shard, "--out_dir", part, "--system_prompt_file", sp,
       "--num_proc", os.environ.get("NPROC", "40"), "--val_size", "1000" if i == 0 else "50"]
sys.exit(subprocess.call(cmd))
