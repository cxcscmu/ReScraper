"""Stitch the N part outputs of tokenize_parallel.sh into the single layout every consumer expects.

Consumers: open_lm reads <dataset_url>/<shard>.tar for each manifest line; pretraining reads num_tokens/size
from the exp_data json. So: move every part's tars into $TOK with a part prefix (part runs all start at 00000001.tar),
rewrite the manifest shard names to match, concatenate, and write one exp_data json whose num_tokens
and size are the sums of the parts'. Idempotent: rerunning after a partial merge is safe.
usage: tokenize_merge.py <corpus> <N>
"""
import os, sys, json, glob, subprocess, shutil, uuid, datetime

U = os.path.dirname(os.path.abspath(__file__))
leg, N = sys.argv[1], int(sys.argv[2])
if leg == "-":   # validation runs: take the layout from the environment
    TOK, READ = os.environ["TOK"], os.environ["READ"]
    CODE = os.environ.get("CODE", os.environ.get("DCLM_DIR", ""))
else:
    env = subprocess.run(["bash", "-c", "source %s/corpora.sh; corpus_vars %s; echo $TOK_OUT; echo $READ; echo $CODE" % (U, leg)],
                         capture_output=True, text=True).stdout.split()
    TOK, READ, CODE = env[0], env[1], env[2]
print("corpus", leg, "TOK", TOK, "READ", READ, flush=True)
lines, ntok, size, meta = [], 0, 0, None
for k in range(N):
    pd = os.path.join(TOK, "p%d" % k)
    mf = os.path.join(pd, "manifest.jsonl")
    js = os.path.join(CODE, "exp_data/datasets/tokenized/%s_p%d.json" % (READ, k))
    assert os.path.exists(mf), "part %d has no manifest: %s" % (k, mf)
    assert os.path.exists(js), "part %d has no exp_data json: %s" % (k, js)
    d = json.load(open(js)); ntok += d["num_tokens"]; size += d["size"]
    if meta is None:
        meta = d
    moved = 0
    for l in open(mf):
        r = json.loads(l)
        src = os.path.join(pd, r["shard"] + ".tar")
        new = "p%d_%s" % (k, r["shard"])
        dst = os.path.join(TOK, new + ".tar")
        if os.path.exists(src):
            os.replace(src, dst); moved += 1
        assert os.path.exists(dst), "missing tar after move: %s" % dst
        lines.append(json.dumps({"shard": new, "num_sequences": r["num_sequences"]}))
    print("part %d: %d shards, %d tokens (moved %d)" % (k, moved if moved else len(lines), d["num_tokens"], moved), flush=True)
tmp = os.path.join(TOK, "manifest.jsonl.tmp")
open(tmp, "w").write("\n".join(lines) + "\n"); os.replace(tmp, os.path.join(TOK, "manifest.jsonl"))
out = dict(meta)
out.update({"uuid": str(uuid.uuid4()), "name": READ, "creation_date": datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            "dataset_url": TOK, "manifest_url": os.path.join(TOK, "manifest.jsonl"), "num_tokens": ntok, "size": size,
            "note": "tokenized in %d parallel parts (tokenize_parallel.sh) and merged by tokenize_merge.py" % N})
jp = os.path.join(CODE, "exp_data/datasets/tokenized/%s.json" % READ)
json.dump(out, open(jp + ".tmp", "w"), indent=4); os.replace(jp + ".tmp", jp)
ntar = len(glob.glob(os.path.join(TOK, "*.tar")))
assert ntar == len(lines), (ntar, len(lines))
for k in range(N):
    pd = os.path.join(TOK, "p%d" % k)
    if os.path.isdir(pd) and not glob.glob(os.path.join(pd, "*.tar")):
        shutil.rmtree(pd)
print("MERGED %s: %d tars, %d tokens (%.3f B), size %d -> %s" % (READ, ntar, ntok, ntok / 1e9, size, jp))
print("TOKENIZE_MERGE_OK")
