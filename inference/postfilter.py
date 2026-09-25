"""Post-filter: drop the pages where the model's first line was a non-standard tag and the executor
passed its raw edit-instruction output through as `text` (typically generations truncated in the middle
of an operation list).

Examples: `<edit>\nrm 10\nrm 29...`, `sub 40: " ######"`, `e2e_tag=<Update>`, `e2e_tag=delete`. The executor
is deliberately NOT changed, so the malformed outputs are removed here instead, between inference and
deduplication (0.061% of documents for the released corpus). Writes a parallel text_clean/ with the same
shard names; text/ is never touched (the dedup step clears its own output dir, and inference may still be
writing).

Drop rules (a doc matching ANY is dropped):
  A2  >=2 lines matching ^(rm N[-N][:N] | sub N: ), or text starts with a bare
      <extract>/<edit>/<delete>/<keep> and is shorter than 200 chars
  A3  e2e_tag not in {<extract>, <refine>, <rewrite>}
  A5  a line that is exactly one of <extract>/<refine>/<rewrite>/<delete>/<edit>
A4 (8-gram repetition > 0.30) is COUNTED ONLY, never dropped: half of it is genuine
source-text repetition (product lists, statutes) and it is what the corpus actually looks like.
It is measured on a 1-in-20 sample because it is diagnostic only and O(words) per doc.

usage: postfilter.py <in_dir> <out_dir> [procs]
"""
import collections
import gzip
import json
import os
import re
import sys

RM_LINE = re.compile(r"^(rm \d+(-\d+)?(:\d+)?|sub \d+: )", re.M)
LEAD_BAD = ("<extract>", "<edit>", "<delete>", "<keep>")
TAG_OK = frozenset(("<extract>", "<refine>", "<rewrite>"))
STANDALONE = frozenset(("<extract>", "<refine>", "<rewrite>", "<delete>", "<edit>"))
REP_SAMPLE = 20
REP_MIN_WORDS = 40
REP_THRESH = 0.30


def judge(text, tag):
    """Return a list of the rules this doc violates (empty = keep)."""
    bad = []
    stripped = text.strip()
    if len(RM_LINE.findall(text)) >= 2 or (
        stripped.startswith(LEAD_BAD) and len(text) < 200
    ):
        bad.append("A2")
    if tag not in TAG_OK:
        bad.append("A3")
    for line in text.split("\n"):
        if line.strip() in STANDALONE:
            bad.append("A5")
            break
    return bad


def repetition(text):
    w = text.lower().split()
    if len(w) < REP_MIN_WORDS:
        return None
    g = [tuple(w[i:i + 8]) for i in range(len(w) - 7)]
    return 1.0 - (len(set(g)) / len(g))


def one_shard(args):
    ind, outd, name = args
    S = collections.Counter()
    dst = os.path.join(outd, name)
    if os.path.exists(dst):
        S["shard_skipped_exists"] += 1
        return S, []
    tmp = "%s.tmp%d" % (dst, os.getpid())
    samples = []
    try:
        with gzip.open(os.path.join(ind, name), "rt", encoding="utf-8") as f, \
                gzip.open(tmp, "wt", encoding="utf-8") as g:
            for i, line in enumerate(f):
                try:
                    r = json.loads(line)
                except ValueError:
                    S["bad_json"] += 1
                    continue
                S["in"] += 1
                text = r.get("text") or ""
                tag = r.get("e2e_tag")
                bad = judge(text, tag)
                if bad:
                    S["dropped"] += 1
                    for b in bad:
                        S["drop_" + b] += 1
                    if len(samples) < 3:
                        samples.append({"shard": name, "rules": bad,
                                        "e2e_tag": tag, "text_head": text[:300]})
                    continue
                if i % REP_SAMPLE == 0:
                    rep = repetition(text)
                    if rep is not None:
                        S["rep_sampled"] += 1
                        if rep > REP_THRESH:
                            S["rep_over_thresh"] += 1
                S["kept"] += 1
                if tag is not None:
                    S["tag_" + str(tag)] += 1
                g.write(json.dumps({"text": text, "e2e_tag": tag},
                                   ensure_ascii=False) + "\n")
    except (OSError, EOFError, gzip.BadGzipFile) as e:
        # Do NOT leave a partial shard behind: a missing output is visible as a count
        # mismatch, a truncated one would silently shrink the corpus.
        if os.path.exists(tmp):
            os.unlink(tmp)
        S.clear()
        S["bad_shard"] += 1
        return S, [{"shard": name, "error": "%s: %s" % (type(e).__name__, e)}]
    os.replace(tmp, dst)
    S["shards"] += 1
    return S, samples


def main():
    ind, outd = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("PROCS", "32"))
    os.makedirs(outd, exist_ok=True)
    names = sorted(f for f in os.listdir(ind) if f.endswith(".jsonl.gz"))
    print("in=%s out=%s shards=%d procs=%d" % (ind, outd, len(names), procs), flush=True)
    T = collections.Counter()
    samples = []
    from multiprocessing import Pool
    with Pool(procs) as p:
        for n, (S, smp) in enumerate(
                p.imap_unordered(one_shard, [(ind, outd, x) for x in names], chunksize=4), 1):
            T.update(S)
            for s in smp:
                if len(samples) < 12:
                    samples.append(s)
            if n % 500 == 0:
                print("  %5d/%d in=%d kept=%d dropped=%d" %
                      (n, len(names), T["in"], T["kept"], T["dropped"]), flush=True)
    print("SAMPLES:", flush=True)
    for s in samples:
        print("  " + json.dumps(s, ensure_ascii=False)[:500], flush=True)
    print("POSTFILTER_STATS %s" % json.dumps(dict(T), ensure_ascii=False), flush=True)
    din = T["in"] or 1
    print("POSTFILTER_DONE in=%d kept=%d dropped=%d (%.4f%%) shards_written=%d bad_shards=%d" %
          (T["in"], T["kept"], T["dropped"], 100.0 * T["dropped"] / din,
           T["shards"], T["bad_shard"]), flush=True)
    if T["bad_shard"]:
        print("POSTFILTER_HAS_BAD_SHARDS", flush=True)
        sys.exit(3)
    print("POSTFILTER_OK", flush=True)


if __name__ == "__main__":
    main()
