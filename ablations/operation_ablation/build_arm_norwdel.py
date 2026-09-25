"""Build the `norwdel` arm (paper: "w/o <rewrite>") = the FULL (control) arm minus every page the model tagged `<rewrite>`.

Why this arm and not `norw`: the `norw` arm of build_arms.py keeps rewrite pages as their extracted text. `<rewrite>`
exists only to RESCUE pages the teacher would otherwise delete, so ablating the operation must send those pages back
to deletion.
The keep-as-extracted arm ends up LARGER than the control (7.63B vs 7.43B tokens) because a RePro paraphrase is only
~77% the length of the extraction it replaces. This arm implements the paper's definition instead.

Rule, in full: for every one of the 10,318 full-arm shards, drop each row whose `e2e_tag` is exactly `"<rewrite>"`.
Every other row is copied BYTE FOR BYTE, in the same order, into the same-named shard. Nothing else changes - no
re-execution, no re-render, no re-read of `$ABL/raw`. The full arm's own files are the only input.

usage: build_arm_norwdel.py build <NT> <TID> [procs]     build the shards with (index % NT == TID)
       build_arm_norwdel.py list                          print the stem count and exit

Per shard it writes:
  $ABL/norwdel/text/<stem>_processed.jsonl.gz   the arm shard (atomic: tmp file + os.replace)
  $ABL/norwdel/meta/<stem>.json                 {full_rows, rw_rows, out_rows, tags, out_sha256, ...}
A shard whose text AND meta exist and whose text sha256 matches the meta is skipped (resumable / requeue-safe).

Self-checks done here per shard:
  B1  out_rows == full_rows - rw_rows
  B2  the sha256 of the concatenated KEPT line bytes computed while reading the full shard equals the sha256 of the
      concatenated line bytes read back from the finished output file, and the row counts agree  -> the written file
      is byte-identical, row by row and in order, to "full minus the <rewrite> rows"
  B3  no row of the output has e2e_tag == "<rewrite>"  (re-parsed from the file on disk, not from memory)
A shard that fails any of them writes meta with "fails" and NO text, and the task exits non-zero.
"""
import gzip, hashlib, json, os, socket, sys, time
from multiprocessing import Pool

R = os.environ["WORK_DIR"]
ABL = os.environ.get("ABL", R + "/dclm_pipeline/operation_ablation")
FULL = ABL + "/full/text"
ARM = ABL + "/norwdel"
OUTD = ARM + "/text"
METAD = ARM + "/meta"
SUF = "_processed.jsonl.gz"
RWLIT = b'"<rewrite>"'
RWTAG = "<rewrite>"


def stems():
    s = sorted(f[: -len(SUF)] for f in os.listdir(FULL) if f.endswith(SUF))
    return s


def build_one(stem):
    t0 = time.time()
    src = os.path.join(FULL, stem + SUF)
    dst = os.path.join(OUTD, stem + SUF)
    mp = os.path.join(METAD, stem + ".json")
    # resume: text + meta present and the sha still matches -> nothing to do
    if os.path.exists(dst) and os.path.exists(mp):
        try:
            m = json.load(open(mp))
            if not m.get("fails") and m.get("out_sha256") == file_sha(dst):
                m["skipped_existing"] = True
                return m
        except Exception:
            pass
    tmp = dst + ".tmp.%d.%d" % (os.getpid(), int(t0))
    full_rows = rw_rows = out_rows = 0
    tags = {}
    keep_h = hashlib.sha256()
    try:
        with gzip.open(src, "rb") as fi, gzip.open(tmp, "wb", compresslevel=6) as fo:
            for line in fi:
                full_rows += 1
                tag = None
                if RWLIT in line:                      # fast path: the literal must appear for the tag to be it
                    try:
                        tag = json.loads(line).get("e2e_tag")
                    except Exception:
                        tag = None
                if tag == RWTAG:
                    rw_rows += 1
                    tags[RWTAG] = tags.get(RWTAG, 0) + 1
                    continue
                keep_h.update(line)
                fo.write(line)
                out_rows += 1
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    fails = []
    if out_rows != full_rows - rw_rows:
        fails.append("B1 out_rows %d != full_rows %d - rw_rows %d" % (out_rows, full_rows, rw_rows))
    # B2/B3: read the FINISHED file back from disk
    back_h = hashlib.sha256()
    back_rows = 0
    back_rw = 0
    with gzip.open(dst, "rb") as g:
        for line in g:
            back_rows += 1
            back_h.update(line)
            if RWLIT in line:
                try:
                    if json.loads(line).get("e2e_tag") == RWTAG:
                        back_rw += 1
                except Exception:
                    pass
    if back_rows != out_rows:
        fails.append("B2 read-back rows %d != written %d" % (back_rows, out_rows))
    if back_h.hexdigest() != keep_h.hexdigest():
        fails.append("B2 read-back sha256 %s != kept-bytes sha256 %s" % (back_h.hexdigest()[:16], keep_h.hexdigest()[:16]))
    if back_rw:
        fails.append("B3 %d <rewrite> rows still in the output" % back_rw)
    meta = {"stem": stem, "full_rows": full_rows, "rw_rows": rw_rows, "out_rows": out_rows,
            "kept_sha256": keep_h.hexdigest(), "out_sha256": file_sha(dst), "out_bytes": os.path.getsize(dst),
            "src_bytes": os.path.getsize(src), "secs": round(time.time() - t0, 2),
            "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID", "-"), "fails": fails}
    if fails:
        os.remove(dst)
        meta["out_sha256"] = None
    tmpm = mp + ".tmp.%d" % os.getpid()
    json.dump(meta, open(tmpm, "w"), indent=1)
    os.replace(tmpm, mp)
    return meta


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    mode = sys.argv[1]
    allst = stems()
    if mode == "list":
        print("STEMS", len(allst))
        return 0
    assert mode == "build", mode
    NT = int(sys.argv[2]); TID = int(sys.argv[3]); PROCS = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    assert 0 <= TID < NT, (NT, TID)
    if len(allst) != 10318:
        print("FATAL full/text has %d shards, expected 10318" % len(allst), flush=True)
        return 2
    os.makedirs(OUTD, exist_ok=True); os.makedirs(METAD, exist_ok=True)
    mine = [s for i, s in enumerate(allst) if i % NT == TID]
    print("[%s] norwdel build NT=%d TID=%d procs=%d shards=%d host=%s" %
          (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), NT, TID, PROCS, len(mine), socket.gethostname()), flush=True)
    tot = {"shards": 0, "full_rows": 0, "rw_rows": 0, "out_rows": 0, "skipped_existing": 0, "failed": 0}
    t0 = time.time()
    with Pool(PROCS) as p:
        for i, m in enumerate(p.imap_unordered(build_one, mine, chunksize=4)):
            tot["shards"] += 1
            tot["full_rows"] += m["full_rows"]; tot["rw_rows"] += m["rw_rows"]; tot["out_rows"] += m["out_rows"]
            if m.get("skipped_existing"):
                tot["skipped_existing"] += 1
            if m["fails"]:
                tot["failed"] += 1
                print("SHARD_FAIL %s %s" % (m["stem"], " | ".join(m["fails"])), flush=True)
            if (i + 1) % 100 == 0 or i + 1 == len(mine):
                print("[%d/%d] %.1f min %s" % (i + 1, len(mine), (time.time() - t0) / 60.0, json.dumps(tot)), flush=True)
    print("TASK_COUNTERS " + json.dumps(tot), flush=True)
    if tot["failed"]:
        print("TASK_FAIL %d shards" % tot["failed"], flush=True)
        return 3
    print("TASK_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
