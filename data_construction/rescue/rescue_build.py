"""Stage-1 selective rescue decisions over the pool, and the FineWeb-Edu sidecar used by both SFT stages.

Pool = $WORK_DIR/dclm_pipeline/two_stage_pool: input/<shard>.jsonl.gz holds Dripper's text of every page and
best/<shard>.jsonl.gz the two-stage refiner's output on it ("" = deleted; ablations/two_stage_refiner/, a 0.6B
model distilled from the same refining teacher, used here as a pool-scale proxy for the teacher's deletions).
For each shard:
  keep as-is                      the pages the two-stage refiner kept
  KEEP the deleted page verbatim  if FineWeb-Edu >= EDU_KEEP (1.5)          <- no rewriting: it damages
                                                                               quizzes, tables, code
  REWRITE the deleted page        if EDU_RW (1.0) <= edu < EDU_KEEP AND the page has a prose block of
                                  >= PROSE_MIN words, and the paraphrase passes a NEW gate:
                                      edu(rewrite) >= edu(original) and edu(rewrite) >= 1.5
                                      0.7 <= len(rewrite)/len(original) <= 1.3
                                      not rwstrict.is_dirty(rewrite)
                                  DataMan is deliberately NOT used: measured pass rate 88.6% on pure
                                  list pages vs 81.9% on prose pages - it rewards fluency, which is what
                                  a rewriter adds to junk.
  drop                            everything else
Outputs: OUT_DIR/<shard>.jsonl.gz (the rescued pages only) and OUT_DIR_decisions/<shard>.jsonl, one row per
deleted page {stem, row, edu, action keep|rewrite|drop[, text]}. The Stage-1 set uses the actions
(sft_sets/build_stage1_set.py); the Stage-2 set uses only the `edu` field (sft_sets/build_stage2_set.py).
env: WORK_DIR, OUT_DIR, EDU_KEEP, EDU_RW, PROSE_MIN, RW_MODEL (the RePro 1B rephraser checkpoint)
usage: rescue_build.py <task_id> <n_tasks>
"""
import os, gzip, json, re, sys, collections
def main():
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
    import rwstrict, pool_join as H, dom_annot as DA
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    R = os.environ["WORK_DIR"]
    DRIP, OURS = R + "/dclm_pipeline/two_stage_pool/input", R + "/dclm_pipeline/two_stage_pool/best"
    OUT = os.environ.get("OUT_DIR", R + "/dclm_pipeline/rescue_add/text")   # ONLY the rescued pages;
    # they get merged into an existing corpus, so the kept pages are not copied here.
    RW_ALL = os.environ.get("RW_ALL", "0") == "1"   # also rewrite the edu>=EDU_KEEP band, to compare keep-vs-rewrite
    EDU_KEEP = float(os.environ.get("EDU_KEEP", "1.5")); EDU_RW = float(os.environ.get("EDU_RW", "1.0"))
    PROSE_MIN = int(os.environ.get("PROSE_MIN", "150"))
    W = re.compile(r"[a-z']+"); PROSE = ("p", "article", "blockquote", "section", "dd", "figcaption")
    tid, nt = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    shards = sorted(f for f in os.listdir(DRIP) if f.endswith(".jsonl.gz"))[tid::nt]
    todo = [f for f in shards if not os.path.exists(os.path.join(OUT, f))]
    print("task %d shards %d todo %d" % (tid, len(shards), len(todo)), flush=True)
    if not todo: return
    dev = "cuda"
    etok = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
    emod = AutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier").to(dev).eval()
    @torch.no_grad()
    def escore(ts):
        out = []
        for i in range(0, len(ts), 64):
            e = etok(ts[i:i+64], return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
            out += emod(**e).logits.squeeze(-1).float().cpu().tolist()
        return out
    from repro_olmo import make_conversation, extract, strip_rules, CHUNK, sampling_params
    from vllm import LLM
    llm = LLM(model=os.environ["RW_MODEL"], dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.55)
    rtok = llm.get_tokenizer()
    S = collections.Counter()
    for f in todo:
        d = [json.loads(l).get("text", "") for l in gzip.open(DRIP + "/" + f, "rt", encoding="utf-8")]
        o = [json.loads(l).get("text", "") for l in gzip.open(OURS + "/" + f, "rt", encoding="utf-8")]
        stem = f.replace(".jsonl.gz", "")
        raw = {}
        p1 = f"{H.ST1}/{stem}.jsonl"
        if os.path.exists(p1):
            s1 = []
            for line in open(p1, encoding="utf-8"):
                try: r1 = json.loads(line)
                except ValueError: continue
                mh = r1.get("main_html") or ""
                if mh.strip(): s1.append((r1.get("input") or "", H.norm_head(mh)))
            i = 0
            for j, t2 in enumerate(d):
                jj = i; found = -1
                while jj < len(s1) and jj < i + 4:
                    if H.head_in(t2, s1[jj][1]): found = jj; break
                    jj += 1
                if found >= 0: raw[j] = s1[found][0]; i = found + 1
        deleted = [j for j in range(min(len(d), len(o))) if d[j].strip() and not o[j].strip()]
        es = escore([d[j] for j in deleted]) if deleted else []
        keep_back, rw_cand = [], []
        for j, s in zip(deleted, es):
            if s >= EDU_KEEP:
                keep_back.append((j, s))
                if RW_ALL: rw_cand.append((j, s))
            elif s >= EDU_RW and j in raw:
                lines = [x.strip() for x in d[j].split("\n") if x.strip()]
                labs = DA.annotate_lines(raw[j], lines)
                run = best = 0
                for ln, lb in zip(lines, labs):
                    if (lb or "").split(".")[0] in PROSE: run += len(W.findall(ln.lower())); best = max(best, run)
                    else: run = 0
                if best >= PROSE_MIN: rw_cand.append((j, s))
        rw_out = {}
        if rw_cand:
            convs, owner = [], []
            for j, s in rw_cand:
                for k in range(0, len(d[j]), CHUNK):
                    convs.append(make_conversation(d[j][k:k+CHUNK], rtok)); owner.append(j)
            outs = llm.chat(convs, sampling_params, use_tqdm=False)
            parts = collections.defaultdict(list)
            for j, ot in zip(owner, outs): parts[j].append(extract(ot.outputs[0].text))
            cand_txt = {j: strip_rules(" ".join(v)) for j, v in parts.items()}
            good = [j for j, t in cand_txt.items() if t.strip()]
            if good:
                rs = escore([cand_txt[j] for j in good])
                src = dict(rw_cand)
                for j, rsc in zip(good, rs):
                    t = cand_txt[j]; lr = len(t.split()) / max(1, len(d[j].split()))
                    if rsc >= src[j] and rsc >= EDU_KEEP and 0.7 <= lr <= 1.3 and not rwstrict.is_dirty(t):
                        rw_out[j] = t; S["rw_pass"] += 1
                    else: S["rw_reject"] += 1
        # sidecar: the per-page decision, so the SFT builder can reuse it instead of rescoring/rewriting
        side = os.path.join(OUT + "_decisions", f.replace(".jsonl.gz", ".jsonl"))
        os.makedirs(os.path.dirname(side), exist_ok=True)
        with open(side + ".tmp%d" % os.getpid(), "w", encoding="utf-8") as sg:
            eduof = dict(zip(deleted, es))
            for j in deleted:
                act = "keep" if any(j == k for k, _ in keep_back) else ("rewrite" if j in rw_out else "drop")
                rec = {"stem": stem, "row": j, "edu": round(float(eduof.get(j, -1)), 3), "action": act}
                if act == "rewrite": rec["text"] = rw_out[j]
                sg.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(side + ".tmp%d" % os.getpid(), side)
        dst = os.path.join(OUT, f); tmp = dst + ".tmp%d" % os.getpid()
        with gzip.open(tmp, "wt", encoding="utf-8") as g:
            for j, s in keep_back:
                rec = {"text": d[j], "rescue": "keep", "edu": round(float(s), 3), "stem": stem, "row": j}
                if RW_ALL and j in rw_out: rec["rewrite"] = rw_out[j]
                g.write(json.dumps(rec, ensure_ascii=False) + "\n"); S["kept_back"] += 1
            for j, tx in rw_out.items():
                if any(j == k for k, _ in keep_back): continue
                g.write(json.dumps({"text": tx, "rescue": "rewrite", "edu": round(float(dict(rw_cand)[j]), 3), "stem": stem, "row": j}, ensure_ascii=False) + "\n")
        os.replace(tmp, dst); S["deleted"] += len(deleted); S["shards"] += 1
        if S["shards"] % 20 == 0: print(tid, dict(S), flush=True)
    print("task", tid, "DONE", dict(S), flush=True)
if __name__ == "__main__":
    main()
