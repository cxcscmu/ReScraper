"""resiliparse extraction + ProX chunk-level refining (ProX-C ONLY, no document-level program) over the full pool
(ProX: arXiv 2409.17115). Produces the ProX-C row of the main table.

Two stages per page, both faithful to their sources:
  1. EXTRACT - resiliparse `extract_plain_text(HTMLTree.parse(html), main_content=True, alt_texts=False,
     preserve_formatting=True)`, i.e. exactly the settings of DCLM's `resiliparse_extraction_modifier`
     (pretraining/dclm_patches/). Input = the SAME page universe the student reads:
     Dripper step1 records with a non-empty `input` (raw html) and a non-empty `main_html` (lib/pool_join.py ST1).
  2. PROX-C - line-numbered chunks of <=1500 tokens, one generation per chunk with
     gair-prox/web-chunk-refining-lm, then the generated program is executed over the document.
     trunc_text / merge_chunks / the sampling params are copied from the ProX repo's
     data_gen/tasks/apply_chunk_refining.py; the executor is that repo's utils/chunk_utils.py verbatim
     (vendored as lib/prox_chunk_utils.py). ProX's document-level model is NOT used, so no
     page is dropped by a doc-level program - a page disappears only when the chunk program's own
     safety thresholds (threshold_2=0.95 / <=10 words) blank it, which is part of the chunk recipe.
     With PROXC_FMT=auto (default) every rank of our pool run selected the "plain" prompt format.

Writes {"text": ...} rows to $INFER_OUT_DIR/<stem>_processed.jsonl.gz, atomic and resumable, so the same
dedup -> tokenize chain as every other corpus can run on it.
usage: proxc_pool.py <rank> <world>
env: INFER_OUT_DIR, PROXC_MODEL, PROXC_FMT, INFER_STEMS_FILE (backfill), LIMIT_FILES, EXTRACT_PROCS, PROXC_DTYPE,
     DRIPPER_STEP1_DIR / WORK_DIR (location of the step-1 records, see lib/pool_join.py)
"""
import os, sys, json, gzip, time, signal, collections

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import pool_join as H
from prox_chunk_utils import execute_meta_operations

OUT = os.environ["INFER_OUT_DIR"]
MODEL = os.environ.get("PROXC_MODEL", "gair-prox/web-chunk-refining-lm")
MAX_LEN = 2048          # the model's max_position_embeddings
MAX_NEW = 256           # ProX apply_chunk_refining.py
CHUNK_TOKENS = 1500     # ProX trunc_text(max_token=1500)
SYS = "You are a helpful, respectful and honest assistant."   # ProX apply_chunk_refining.py
FMT = os.environ.get("PROXC_FMT", "auto")
rank, world = int(sys.argv[1]), int(sys.argv[2])

if os.environ.get("INFER_STEMS_FILE"):
    allst = [l.strip() for l in open(os.environ["INFER_STEMS_FILE"]) if l.strip()]
else:
    allst = sorted(f[:-6] for f in os.listdir(H.ST1) if f.endswith(".jsonl"))
stems = allst[rank::world]
lim = int(os.environ.get("LIMIT_FILES", "0"))
stems = stems[:lim] if lim else stems
os.makedirs(OUT, exist_ok=True)
todo = [s for s in stems if not os.path.exists(os.path.join(OUT, s + "_processed.jsonl.gz"))]
print("rank %d/%d shards %d todo %d model %s" % (rank, world, len(stems), len(todo), MODEL), flush=True)
if not todo:
    sys.exit(0)

# ---------- stage 1: resiliparse, in worker processes with a per-page alarm ----------
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.html import HTMLTree


class TO(Exception):
    pass


def _init():
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))


def extract(html, sec=60):
    signal.alarm(sec)
    try:
        return extract_plain_text(HTMLTree.parse(html), main_content=True, alt_texts=False,
                                  preserve_formatting=True)
    except Exception:
        return None
    finally:
        signal.alarm(0)


from multiprocessing import Pool
pool = Pool(int(os.environ.get("EXTRACT_PROCS", "10")), initializer=_init)

# ---------- stage 2: ProX-C ----------
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, dtype=os.environ.get("PROXC_DTYPE", "auto"), max_model_len=MAX_LEN,
          gpu_memory_utilization=0.85, enable_prefix_caching=False)
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=MAX_NEW)   # ProX apply_chunk_refining.py


def trunc_text(text, max_token=CHUNK_TOKENS, max_digits=3):
    """ProX trunc_text, with the per-line encode done in one batched call (same counts, ~100x faster)."""
    lines = text.split("\n")
    norm = ["[%0*d]%s" % (max_digits, i, l) for i, l in enumerate(lines)]
    counts = [len(x) for x in tok(norm)["input_ids"]]   # default add_special_tokens, as tokenizer.encode
    chunks, cur, cur_n = [], [], 0
    for normalize_line, line_token_count in zip(norm, counts):
        if cur_n + line_token_count <= max_token:
            cur.append(normalize_line); cur_n += line_token_count
        else:
            if cur:
                chunks.append("\n".join(cur))
            cur = [normalize_line]; cur_n = line_token_count
            if line_token_count > max_token:
                chunks.append("\n".join(cur)); cur = []; cur_n = 0
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def wrap(fmt, chunk):
    """ProX calls tokenizer.apply_chat_template, but this checkpoint ships NO chat_template: their
    transformers was old enough to fall back to LlamaTokenizer.default_chat_template, i.e. the Llama-2
    [INST]/<<SYS>> format (which is also where that exact system sentence comes from). That fallback no
    longer exists, so the candidates are written out here and PROXC_FMT=auto picks between them on real
    pages before the run starts - a wrong template would silently produce a pool-sized pile of no-ops."""
    u = "[doc]\n%s\n[/doc]" % chunk
    if fmt == "llama2_nosys":
        return "<s>[INST] %s [/INST]" % u.strip()
    if fmt == "plain":
        return u
    if fmt == "chatml":
        return "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % (SYS, u)
    return "<s>[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]" % (SYS, u.strip())


CANDS = ["llama2_sys", "llama2_nosys", "plain", "chatml"]
OPS = ("remove_lines", "normalize")


def pick_format(sample_docs):
    """Score each candidate on real pages: a template the model was trained on emits parsable ProX programs
    on most documents and keeps a sane share of the words. Anything that emits no ops at all, or blanks
    nearly everything, is the wrong template. Deterministic (greedy), so every rank picks the same one."""
    best = (None, -1.0)
    for fmt in CANDS:
        prompts, lens = [], []
        for d in sample_docs:
            cs = trunc_text(d); lens.append(len(cs))
            prompts.extend(wrap(fmt, c) for c in cs)
        outs = [o.outputs[0].text.strip(" ") for o in llm.generate(prompts, sp, use_tqdm=False)]
        progs, i = [], 0
        for n in lens:
            progs.append("\n".join(outs[i:i + n])); i += n
        ops = sum(1 for p in progs if any(l.strip().startswith(OPS) for l in p.split("\n") if l.strip()))
        win = wout = 0; kept = 0
        for d, p in zip(sample_docs, progs):
            try:
                new = execute_meta_operations(text=d, operations=p, threshold_1=0.0, threshold_2=0.95, error_op=2)
            except Exception:
                new = ""
            if new.strip():
                kept += 1; win += len(d.split()); wout += len(new.split())
        frac_ops = ops / max(len(progs), 1); frac_kept = kept / max(len(sample_docs), 1)
        wr = wout / max(win, 1)
        ok = frac_ops >= 0.5 and frac_kept >= 0.5 and 0.30 <= wr <= 0.999
        print("  fmt %-13s ops %.2f kept %.2f words_kept %.3f -> %s" % (fmt, frac_ops, frac_kept, wr, "ok" if ok else "no"), flush=True)
        if ok and frac_ops > best[1]:
            best = (fmt, frac_ops)
    return best[0]


def sample_docs(stem, want=8):
    out = []
    with open("%s/%s.jsonl" % (H.ST1, stem), encoding="utf-8") as g:
        for line in g:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not (r.get("input") or "").strip() or not (r.get("main_html") or "").strip():
                continue
            t = extract(r["input"])
            if t and len(t.split()) > 60:
                out.append(t)
            if len(out) >= want:
                break
    return out


if FMT == "auto":
    # A fixed, global reference stem: todo[0] differs per rank (stems = allst[rank::world]) and also
    # drifts across requeues as finished shards leave todo, so ranks could pick different templates
    # and build one corpus under several prompts. listdir(ST1) is identical everywhere and stable.
    REFSTEM = sorted(f[:-6] for f in os.listdir(H.ST1) if f.endswith(".jsonl"))[0]
    print("rank %d picking prompt format on real pages of %s (fixed reference stem)" % (rank, REFSTEM), flush=True)
    FMT = pick_format(sample_docs(REFSTEM))
    if FMT is None:
        print("FATAL rank %d: no candidate prompt format produced usable ProX programs - refusing to write a corpus" % rank, flush=True)
        sys.exit(2)
    print("rank %d FORMAT=%s" % (rank, FMT), flush=True)

S = collections.Counter()
for stem in todo:
    t0 = time.time(); raws = []
    with open("%s/%s.jsonl" % (H.ST1, stem), encoding="utf-8") as g:
        for line in g:
            try:
                r1 = json.loads(line)
            except ValueError:
                S["bad_step1_json"] += 1; continue
            raw = r1.get("input") or ""
            if not raw.strip():
                S["empty_step1_input"] += 1; continue
            if not (r1.get("main_html") or "").strip():
                S["no_main_html"] += 1; continue
            raws.append(raw)
    texts = pool.map(extract, raws, chunksize=16)

    docs, prompts, chunk_lens = [], [], []
    for txt in texts:
        if not txt or not txt.strip():
            S["extract_fail_or_empty"] += 1; continue
        ch = trunc_text(txt)
        keep = []
        for c in ch:
            p = wrap(FMT, c)
            if len(tok(p)["input_ids"]) > MAX_LEN - MAX_NEW:
                S["chunk_too_long_skipped"] += 1   # a single line longer than the chunk budget
                continue
            keep.append(p)
        docs.append(txt); chunk_lens.append(len(keep)); prompts.extend(keep)
    S["docs"] += len(docs); S["chunks"] += len(prompts)

    outs = llm.generate(prompts, sp, use_tqdm=False) if prompts else []
    outs = [o.outputs[0].text.strip(" ") for o in outs]
    programs, i = [], 0
    for n in chunk_lens:                      # ProX merge_chunks
        programs.append("\n".join(outs[i:i + n])); i += n

    dst = os.path.join(OUT, stem + "_processed.jsonl.gz"); tmp = dst + ".tmp%d" % os.getpid(); n_out = 0
    win = wout = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as g:
        for text, program in zip(docs, programs):
            try:
                new = execute_meta_operations(text=text, operations=program,
                                              threshold_1=0.0, threshold_2=0.95, error_op=2)
            except Exception:
                S["execute_fail"] += 1; continue
            if not new or not new.strip():
                S["blanked_by_program"] += 1; continue
            if new == text:
                S["unchanged"] += 1
            else:
                S["edited"] += 1
            win += len(text.split()); wout += len(new.split())
            g.write(json.dumps({"text": new}, ensure_ascii=False) + "\n"); n_out += 1
    os.replace(tmp, dst)
    S["pages"] += len(raws); S["written"] += n_out; S["words_in"] += win; S["words_out"] += wout
    print("rank %d %s: pages %d kept %d in %.0fs | %s" % (rank, stem, len(raws), n_out, time.time() - t0, dict(S)), flush=True)
pool.close(); pool.join()
print("rank %d DONE %s" % (rank, dict(S)), flush=True)
