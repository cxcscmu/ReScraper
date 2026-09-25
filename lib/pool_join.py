"""Locations of the Dripper outputs over the source pool and the helpers used to join rows across them.

ST1 = Dripper step 1 (main-content inference): one `<stem>.jsonl` per pool shard, one row per page, with
      `input` (the raw HTML) and `main_html` (the region Dripper selected; empty when it found none).
ST2 = Dripper step 2 (main_html -> text): one `<stem>.jsonl.gz` per shard, rows in step-1 order but with the
      pages whose conversion failed dropped. Rows are re-aligned to step 1 by `head_in(text, norm_head(main_html))`,
      an ordered check that the first words of the extracted text occur, in order, in the head of main_html.

Both directories default to `$WORK_DIR/dclm_pipeline/dripper/{step1_inference,step2_text}` and can be overridden
with DRIPPER_STEP1_DIR / DRIPPER_STEP2_DIR. See data_construction/dripper/ for how they are produced.
"""
import gzip, hashlib, html as H, json, os, re

_WORK = os.environ.get("WORK_DIR", "")
ST1 = os.environ.get("DRIPPER_STEP1_DIR", os.path.join(_WORK, "dclm_pipeline", "dripper", "step1_inference"))
ST2 = os.environ.get("DRIPPER_STEP2_DIR", os.path.join(_WORK, "dclm_pipeline", "dripper", "step2_text"))

TAG = re.compile(r"<[^>]+>"); WORD = re.compile(r"[a-z0-9]{3,}")
md5 = lambda s: hashlib.md5(s.encode("utf-8", "surrogatepass")).hexdigest()


def norm_head(h, n=600):
    return " ".join(H.unescape(TAG.sub(" ", h[:20000])).split()).lower()[:n]


def head_in(text, head):
    ws = WORD.findall(text[:200].lower())[:6]
    if not ws: return False
    pos = 0
    for w in ws:
        j = head.find(w, pos)
        if j < 0: return False
        pos = j + len(w)
    return True


def gz_rows(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for l in f: yield json.loads(l)
