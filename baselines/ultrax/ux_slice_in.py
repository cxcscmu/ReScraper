"""One array task = one slice of the parquet shards ux_pool_prep.py wrote.

UltraX's inference.py takes a directory, not a file list, and skips any input whose output parquet
already exists - so each task gets its own directory of symlinks (names preserved, which is what makes
the shard identity survive) and every task writes into the SAME output dir. Striped, not contiguous,
so a slice is a mix of the pool rather than one region of it. Re-running is free: existing links stay.
usage: ux_slice_in.py <task_id> <n_tasks>   env: UX_WORK (or WORK_DIR)
"""
import os, sys

R = os.environ.get("WORK_DIR", "")
W = os.environ.get("UX_WORK", os.path.join(R, "dclm_pipeline", "resiliparse_raw_ultrax"))
IN = W + "/in"
tid, nt = int(sys.argv[1]), int(sys.argv[2])
d = "%s/in_parts/%03d" % (W, tid)
os.makedirs(d, exist_ok=True)
files = sorted(f for f in os.listdir(IN) if f.endswith(".parquet"))[tid::nt]
new = 0
for f in files:
    link = os.path.join(d, f)
    if not os.path.lexists(link):
        os.symlink(os.path.join(IN, f), link)
        new += 1
print("slice %d/%d: %d shards (%d new links) -> %s" % (tid, nt, len(files), new, d), flush=True)
