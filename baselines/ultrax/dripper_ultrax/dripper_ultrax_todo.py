#!/usr/bin/env python
"""Build a todo symlink dir for one Dripper -> UltraX part: inputs whose inference output
parquet is missing or corrupt (validates footer; deletes corrupt outputs so the
run regenerates them). Wipes and rebuilds the todo dir each call.
usage: dripper_ultrax_todo.py --inp <in/partK> --out <inf/partK> --todo <todo dir> [--limit N --mod M --rem R]"""
import argparse, glob, os, shutil

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--todo", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--mod", type=int, default=1)
ap.add_argument("--rem", type=int, default=0)
a = ap.parse_args()

shutil.rmtree(a.todo, ignore_errors=True)
os.makedirs(a.todo)
os.makedirs(a.out, exist_ok=True)
n_todo = n_done = n_bad = 0
for idx, f in enumerate(sorted(glob.glob(a.inp + "/*.parquet"))):
    if idx % a.mod != a.rem:
        continue
    out = os.path.join(a.out, os.path.basename(f))
    if os.path.exists(out):
        try:
            pq.ParquetFile(out).metadata
            n_done += 1
            continue
        except Exception:
            os.remove(out)
            n_bad += 1
    if a.limit and n_todo >= a.limit:
        continue
    os.symlink(f, os.path.join(a.todo, os.path.basename(f)))
    n_todo += 1
print(f"todo={n_todo} done={n_done} corrupt_removed={n_bad}")
