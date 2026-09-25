"""noext arm (paper: "w/o <extract>"): run the released model's program WITHOUT the model's own <extract> block.

Built from the SAME raw records as the other arms (one T=1 full-pool pass, infer_pool_with_raw.py). Derived from
build_arms.py: same row writer, same F3 drop, same full-arm rebuild.

usage: build_arm_noext.py <ABL dir> <OUT dir> <task> <ntasks> <procs> [--g1]      shards[task::ntasks]
  shards = sorted(raw stems that also have full/text)[task::ntasks]   (ntasks passed explicitly, never from SLURM)
  (our run distributed the same per-shard work over workers that claimed shards dynamically; the per-shard output
  is deterministic, so the static split produces identical shards)
reads  <ABL>/raw/<stem>_raw.jsonl.gz, <ABL>/full/text/<stem>_processed.jsonl.gz (post-drop), step1 <stem>.jsonl (input)
writes <OUT>/text/<stem>_processed.jsonl.gz and <OUT>/meta/<stem>.json (atomic; a shard whose text + meta exist and
       whose text sha256 == meta sha256 is skipped = resumable per shard)

Rule per generated page (the program is executed by a line-for-line mirror of rescraper_ops.body_from_prediction_dfirst,
including its staged fallback, with ops1 = the <extract> block replaced by nothing):
  <keep>   -> apply_ops(ninp, "")          = the full rendering, lid markers stripped (executor's own apply_ops)
  <edit>   -> ops payload: _apply_edit_body(ninp, "", rest) = apply_ops(ninp, ops2) (edit ops only, original lids)
              non-op payload (full-text fallback): payload unchanged (counted: path_edit_fallback)
  <rewrite>, <delete>, other tags, parse_fail, F3 pages: exactly as in the full arm (F3 = <extract> with final !=
              extracted, the 40 pages build_arms.py dropped; dropped here too)
  row emitted by the writer's own rule: tag != <delete> and text non-empty. keep/edit-op rows (the pages whose text
  the rule changes) are written {"text", "e2e_tag": <original tag>, "ablation_from": <original tag>}; every other row is
  the full-arm row byte for byte.
ninp is not in production raw, so it is re-rendered per page from step1 `input` with the inference script's own
renderer (copied verbatim from infer_pool_with_raw.py, same dripper patch, same number_lines).
Checks:
  G1 (--g1, smoke raw carries ninp): re-rendered ninp == saved ninp, byte for byte, every generated page (fatal)
  G2 every generated page: extracted_of(ninp_rerendered, gen) == saved extracted AND the UNMODIFIED executor
     body_from_prediction_dfirst(ninp_rerendered, gen) == saved (tag, final). A page failing G2 (or whose step1
     md5 differs / re-render fails) falls back to its full-arm row and is counted (gate: fatal above 0.1%).
  MIRROR the mirror with the real ops1 == the executor on every G2 page (fatal: the rule would be mis-executed)
  F1 full rows rebuilt from raw (drop rule) == full/text on disk (fatal)
  G3 per shard: every full-arm page emits a noext row (fatal otherwise); noext pages not in full are recorded as
     revived (keep/edit pages whose full text was empty only because of their <extract> ops)
"""
import collections, gzip, hashlib, json, os, re, resource, signal, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import rescraper_ops as X, editops as E, pool_join

VERSION = "noext-v2"
ST1 = pool_join.ST1
RENDER_SEC = int(os.environ.get("NOEXT_RENDER_SEC", "300"))
CF_PATHS = ("keep", "edit_ops")
# v2: only pages whose noext text can depend on ninp are re-rendered: saved tag <extract> (keep path, or the F3/other
# path) or <refine> (edit paths). <rewrite>/<delete>/other tags/parse_fail never touch ninp in the executor, so their
# noext row is the full-arm row by construction; 1 in G2_SAMPLE of them (idx % G2_SAMPLE == 0) is re-rendered anyway so
# that G2 is also measured on every tag. (v1 rendered every page: 850 core-h at the smoke's 0.17 s/page.)
NEED_TAGS = ("<extract>", "<refine>")
G2_SAMPLE = int(os.environ.get("NOEXT_G2_SAMPLE", "20"))
STD = ("<extract>", "<refine>", "<rewrite>", "<delete>")
assert not X.OPS_TOLERANT, "OPS_TOLERANT must be unset (release default)"

# ---- renderer: verbatim from infer_pool_with_raw.py ----------------------------------------------------
import dripper.process.simplify_html as sh
sh.tags_to_remove = {"head", "style", "script", "noscript", "link", "meta", "iframe", "frame"}; sh.ATTR_PATTERNS_TO_REMOVE = set()
from webpage_converter.convert import convert_html_to_structured_data
from bs4 import BeautifulSoup
def webkit_txt(html):
    raw = convert_html_to_structured_data(html, output_format="txt"); text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    text = re.sub(r"\|+", " ", text); text = re.sub(r"^[\s\-\|]+$", "", text, flags=re.M); text = re.sub(r"\\([\$\*\[\]_`#])", r"\1", text)
    return "\n".join(l for l in (re.sub(r"\s+", " ", x).strip() for x in text.split("\n")) if l)
class TO(Exception): pass
def _init(): signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))
def render(h, sec=RENDER_SEC):
    signal.alarm(sec)
    try: return webkit_txt(h)
    except Exception: return None
    finally: signal.alarm(0)
def extracted_of(ninp, pred):
    lines = pred.strip().split("\n")
    d = lines[0].strip() if lines else ""
    ops1 = X.parse_staged("\n".join(lines[1:]))[0] if d in X.DFIRST_TAGS else X.parse_staged(pred)[0]
    return E.apply_ops(ninp, ops1)
def md5(s): return hashlib.md5(s.encode("utf-8", "surrogatepass")).hexdigest()
# -----------------------------------------------------------------------------------------------------------------


def run_program(ninp, pred, drop_ext):
    """Line-for-line mirror of rescraper_ops.body_from_prediction_dfirst (+ body_from_prediction_staged) that also returns
    the executor path. drop_ext=True executes the same program with ops1 (the <extract> block) = ""."""
    lines = pred.strip().split("\n")
    d = lines[0].strip() if lines else ""
    if d not in X.DFIRST_TAGS:
        ops1, tag, rest = X.parse_staged(pred)
        if drop_ext: ops1 = ""
        if tag == "<delete>": return "<delete>", "", "delete", True
        if tag == "<rewrite>": return "<rewrite>", rest.strip(), "rewrite", True
        if tag == "<keep>": return "<extract>", E.apply_ops(ninp, ops1), "keep", True
        if tag == "<edit>":
            t, as_ops = X._apply_edit_body(ninp, ops1, rest)
            return "<refine>", t, ("edit_ops" if as_ops else "edit_fallback"), True
        return tag, rest.strip(), "other", True
    ops1, tag2, rest = X.parse_staged("\n".join(lines[1:]))
    if drop_ext: ops1 = ""
    if d == "<delete>": return "<delete>", "", "delete", False
    if d == "<keep>": return "<extract>", E.apply_ops(ninp, ops1), "keep", False
    if tag2 not in X.DFIRST_TAGS:
        rest = (tag2 + "\n" + rest) if tag2 else rest
    if d == "<rewrite>": return "<rewrite>", rest.strip(), "rewrite", False
    t, as_ops = X._apply_edit_body(ninp, ops1, rest)
    return "<refine>", t, ("edit_ops" if as_ops else "edit_fallback"), False


def row(text, tag, src=None):
    d = {"text": text, "e2e_tag": tag}
    if src is not None:
        d["ablation_from"] = src
    return json.dumps(d, ensure_ascii=False) + "\n"


def read_lines(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return f.readlines()


def sha_text(p):
    h = hashlib.sha256()
    with gzip.open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic_gz(p, lines):
    tmp = "%s.tmp%s.%d" % (p, os.uname().nodename.split(".")[0], os.getpid())
    with gzip.open(tmp, "wt", encoding="utf-8") as g:
        g.writelines(lines)
    os.replace(tmp, p)


def write_atomic_json(p, obj):
    tmp = "%s.tmp%s.%d" % (p, os.uname().nodename.split(".")[0], os.getpid())
    with open(tmp, "w") as g:
        json.dump(obj, g)
    os.replace(tmp, p)


def shard_done(out, stem):
    dst, mp = f"{out}/text/{stem}_processed.jsonl.gz", f"{out}/meta/{stem}.json"
    if not (os.path.exists(dst) and os.path.exists(mp)):
        return None
    try:
        m = json.load(open(mp))
        if m.get("version") == VERSION and not m.get("fails") and m.get("sha256") == sha_text(dst):
            return m
    except Exception:
        pass
    return None


EMPTY_META = {"counters": {}, "rows": 0}


def one(args):
    abl, out, stem, g1 = args[:4]
    m = shard_done(out, stem)
    if m is not None:
        return stem, "skip", m
    t0 = time.time()
    S = collections.Counter(); fails = []
    recs = [json.loads(l) for l in read_lines(f"{abl}/raw/{stem}_raw.jsonl.gz")]
    full_disk = read_lines(f"{abl}/full/text/{stem}_processed.jsonl.gz")
    if [r["idx"] for r in recs] != sorted(r["idx"] for r in recs) or len({r["idx"] for r in recs}) != len(recs):
        fails.append(f"{stem}: raw idx not strictly increasing")
    need = {r["idx"]: r for r in recs if "skip" not in r and (r.get("tag") in NEED_TAGS or r["idx"] % G2_SAMPLE == 0)}
    ninps = {}
    with open(f"{ST1}/{stem}.jsonl", encoding="utf-8") as g:
        for idx, line in enumerate(g):
            r = need.get(idx)
            if r is None:
                continue
            try:
                r1 = json.loads(line)
            except ValueError:
                S["rr_bad_json"] += 1; continue
            raw = r1.get("input") or ""
            if md5(raw) != r.get("in_md5"):
                S["rr_md5_mismatch"] += 1; continue
            txt = render(raw)
            if not txt or len(txt) < 20:
                S["rr_render_fail"] += 1; continue
            ninps[idx] = E.number_lines(txt)
    t_render = time.time() - t0
    full_rows, full_idx, rows, rows_idx = [], [], [], []
    revived, g2_mm, fb_edit = [], [], []
    for r in recs:
        if "skip" in r:
            S["skip_" + r["skip"]] += 1; continue
        S["generated"] += 1
        idx, gen, tag, final, ext = r["idx"], r["gen"], r.get("tag"), r.get("final"), r.get("extracted")
        ninp = ninps.get(idx); selected = idx in need
        if g1 and selected:
            if ninp is not None and ninp == r.get("ninp"): S["g1_ok"] += 1
            else:
                S["g1_mismatch"] += 1; fails.append(f"{stem}:{idx} G1 re-rendered ninp != saved ninp")
        if not selected:
            S["g2_not_rendered"] += 1
            if tag in NEED_TAGS:
                fails.append(f"{stem}:{idx} page with tag {tag} was not rendered"); continue
        ok2 = ninp is not None
        if not selected:
            pass
        elif ok2:
            try: xo = extracted_of(ninp, gen)
            except Exception: xo = None
            try: ex, ex_err = X.body_from_prediction_dfirst(ninp, gen), False
            except Exception: ex, ex_err = None, True
            g2a = xo == ext
            g2b = (ex_err and tag is None) or (not ex_err and tag is not None and ex == (tag, final))
            S["g2a_ok"] += int(g2a); S["g2b_ok"] += int(g2b)
            ok2 = g2a and g2b
        else:
            S["g2_no_ninp"] += 1
        if selected:
            S["g2_checked"] += 1
            if ok2: S["g2_ok"] += 1
            else:
                S["g2_mismatch"] += 1; g2_mm.append(idx)
        if tag is None:
            S["parse_fail"] += 1; continue
        S["tag" + (tag if tag in STD else "_other")] += 1
        # full arm exactly as build_arms.py rebuilds it
        ext_d = ext if ext is not None else ""
        in_full = not (tag == "<delete>" or not (final or "").strip())
        if tag == "<extract>" and final != ext_d:
            S["f3_pages"] += 1; S["f3_dropped_full_rows"] += int(in_full); continue
        if in_full:
            full_rows.append(row(final, tag)); full_idx.append(idx)
        # noext
        if not selected:
            nx, cf, path = final, False, "not_rendered"          # tag outside NEED_TAGS: executor never reads ninp
            S["path_not_rendered_" + (tag if tag in STD else "other")] += 1
        elif ok2:
            t_m, x_m, p_m, st_m = run_program(ninp, gen, False)
            if (t_m, x_m) != (tag, final):
                fails.append(f"{stem}:{idx} MIRROR differs from the executor"); continue
            nt, nx, path, staged = run_program(ninp, gen, True)
            if nt != tag or path != p_m:
                fails.append(f"{stem}:{idx} noext changed the tag/path {tag}/{p_m} -> {nt}/{path}"); continue
            if path not in CF_PATHS and nx != final:
                fails.append(f"{stem}:{idx} non-extract path {path} changed the text"); continue
            S["path_" + path] += 1
            if staged: S["staged_path_" + path] += 1
            if path == "edit_fallback":
                fb_edit.append(idx); S["fallback_edit_chars"] += len(nx or "")
            cf = path in CF_PATHS
        else:
            nx, cf, path = final, False, "g2_fallback"
            S["path_g2_fallback_full_text"] += 1
        emit = not (tag == "<delete>" or not (nx or "").strip())
        if emit:
            rows.append(row(nx, tag, tag) if cf else row(final, tag)); rows_idx.append(idx)
            S["noext_rows"] += 1; S["noext_chars"] += len(nx)
            S["noext_rows_tag" + (tag if tag in STD else "_other")] += 1
            if cf:
                S["cf_rows"] += 1; S["cf_rows_" + path] += 1
                S["cf_chars_" + path] += len(nx); S["cf_full_chars_" + path] += len(final) if in_full else 0
                if nx == final: S["cf_rows_unchanged_text"] += 1
            if not in_full:
                revived.append(idx); S["revived_rows"] += 1; S["revived_rows_" + path] += 1; S["revived_chars"] += len(nx)
        elif in_full:
            fails.append(f"{stem}:{idx} G3 page is in the full arm but emits no noext row")
    S["full_rows"] += len(full_rows); S["full_chars"] += sum(len(json.loads(l)["text"]) for l in full_rows)
    if full_rows != full_disk:
        fails.append(f"{stem}: F1 full rows rebuilt from raw ({len(full_rows)}) != post-drop full/text on disk ({len(full_disk)})")
    S["shards"] += 1
    meta = {"version": VERSION, "stem": stem, "fails": fails[:50], "n_fail": len(fails), "counters": dict(S),
            "rows": len(rows), "rows_idx": rows_idx, "full_idx": full_idx, "revived_idx": revived,
            "g2_mismatch_idx": g2_mm, "fallback_edit_idx": fb_edit,
            "secs": round(time.time() - t0, 1), "render_secs": round(t_render, 1), "rendered": len(ninps)}
    os.makedirs(f"{out}/meta", exist_ok=True)
    dst = f"{out}/text/{stem}_processed.jsonl.gz"
    if not fails:
        write_atomic_gz(dst, rows)
        meta["sha256"] = sha_text(dst)
    write_atomic_json(f"{out}/meta/{stem}.json", meta)
    return stem, ("fail" if fails else "built"), meta


def run_pool(procs, items, T, st, t0, total):
    from multiprocessing import Pool
    nf = 0
    with Pool(procs, initializer=_init) as p:
        for n, (stem, status, m) in enumerate(p.imap_unordered(one, items, chunksize=1), 1):
            st[status] += 1; T.update(m["counters"])
            if status == "fail":
                nf += 1
                for f in m["fails"][:5]:
                    print("FAIL " + f, flush=True)
            if status in ("built", "fail") or n % 500 == 0:
                c = m["counters"]
                print(f"  {n}/{total} {status} {stem} gen={c.get('generated', 0)} rows={m['rows']} "
                      f"g2_mm={c.get('g2_mismatch', 0)} revived={c.get('revived_rows', 0)} secs={m.get('secs')} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
    return nf


def main():
    abl, out = sys.argv[1], sys.argv[2]
    g1 = "--g1" in sys.argv
    os.makedirs(f"{out}/text", exist_ok=True); os.makedirs(f"{out}/meta", exist_ok=True)
    stems = sorted(f[:-len("_raw.jsonl.gz")] for f in os.listdir(f"{abl}/raw") if f.endswith("_raw.jsonl.gz"))
    have_full = {f[:-len("_processed.jsonl.gz")] for f in os.listdir(f"{abl}/full/text") if f.endswith("_processed.jsonl.gz")}
    allst = [s for s in stems if s in have_full]
    T = collections.Counter(); st = collections.Counter(); t0 = time.time()
    task, ntasks, procs = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    mine = allst[task::ntasks]
    mine.sort(key=lambda s: -os.path.getsize(f"{abl}/raw/{s}_raw.jsonl.gz"))  # largest first (load balance)
    print(f"[{time.strftime('%F %T', time.gmtime())} UTC] noext task {task}/{ntasks} abl={abl} out={out} raw={len(stems)} "
          f"full={len(have_full)} mine={len(mine)} procs={procs} g1={g1} render_sec={RENDER_SEC} version={VERSION}", flush=True)
    nf = run_pool(procs, [(abl, out, s, g1) for s in mine], T, st, t0, len(mine))
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    print(f"TASK_COUNTERS {json.dumps(dict(T))}", flush=True)
    print(f"TASK_DONE status={dict(st)} failed={nf} elapsed={time.time() - t0:.0f}s max_child_rss_GB={rss:.2f}", flush=True)
    if nf:
        print(f"TASK_FAIL {nf} shards failed", flush=True); sys.exit(3)
    print("TASK_OK", flush=True)


if __name__ == "__main__":
    main()
