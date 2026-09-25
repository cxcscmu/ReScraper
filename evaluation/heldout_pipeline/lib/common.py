"""Shared helpers for the page-aligned held-out pipeline (evaluation/heldout_pipeline).

Every per-page file in this pipeline is JSONL keyed by `gid` (the row id of the page table).
The page table (heldout5k.jsonl, built by evaluation/heldout_set/) has
    gid, stem, step1_idx, in_md5, url, warc_id, input, drip, output, tag, gfinal, edu, resiliparse
Work dirs: $HELDOUT_PIPELINE_DIR/work/<tag> (tag = page-table file name without .jsonl; override with HP_WORK).
Figure data: $HELDOUT_PIPELINE_DIR/data (heldout5k) or <work>/data (any other table); override with HP_OUT.
"""
import gzip, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(EVAL, ".."))
LIB_E2E = os.path.join(REPO, "lib")                   # e2e_ops.py + editops.py (the executor of the pool writer)
if EVAL not in sys.path:
    sys.path.insert(0, EVAL)
from eval_paths import HELDOUT_PIPELINE_DIR as HP      # noqa: E402
SYS_PROMPT_FILE = os.path.join(REPO, "prompts", "student_system_stage2.txt")   # system prompt of the released model

# system keys used in every output of this pipeline (same keys as the paper's data files)
SYSTEMS = ["rescraper", "ultrax", "proxc", "refinedweb_rule", "fineweb_rule"]


def tag_of(page_table):
    """Work-dir tag derived from the page table path: .../heldout5k/heldout5k.jsonl -> heldout5k."""
    b = os.path.basename(page_table)
    for suf in (".jsonl.gz", ".jsonl"):
        if b.endswith(suf):
            b = b[: -len(suf)]
    return b


def workdir(page_table):
    d = os.environ.get("HP_WORK") or os.path.join(HP, "work", tag_of(page_table))
    os.makedirs(d, exist_ok=True)
    return d


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_pages(path, fields=None):
    rows = []
    for r in read_jsonl(path):
        rows.append({k: r.get(k) for k in fields} if fields else r)
    gids = [r["gid"] for r in rows]
    assert gids == list(range(len(rows))), "page table must be gid 0..N-1 in order"
    return rows


def load_by_gid(path):
    out = {}
    for r in read_jsonl(path):
        assert r["gid"] not in out, ("duplicate gid", path, r["gid"])
        out[r["gid"]] = r
    return out


def write_jsonl_atomic(path, rows):
    tmp = path + ".tmp"
    op = gzip.open if path.endswith(".gz") else open
    with op(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def md5(s):
    return hashlib.md5(s.encode("utf-8", "surrogatepass")).hexdigest()


def nonempty(t):
    return isinstance(t, str) and bool(t.strip())


# teacher / student decision vocabularies -> the four operation names used in the paper
OPNAME = {"<extract>": "keep", "<keep>": "keep", "<refine>": "edit", "<edit>": "edit",
          "<delete>": "delete", "<rewrite>": "rewrite"}


def e2e():
    """(e2e_ops, editops) modules from lib/."""
    if LIB_E2E not in sys.path:
        sys.path.insert(0, LIB_E2E)
    import e2e_ops as X, editops as E
    return X, E
