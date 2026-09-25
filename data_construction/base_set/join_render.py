"""Raw HTML and the canonical rendering for the selected seed pages: step2 text row -> step1 record (ordered,
head check, lib/pool_join.py) -> step1.input = raw HTML. The candidate is accepted only if webkit_txt(main_html)
reproduces the step2 text exactly. Emits rows {i, stem, row, tag, input = webkit_txt(raw page) (the model input
before line numbering), drip = Dripper text, output = teacher target}. Pages whose rendering fails, times out
(60 s) or is shorter than 20 characters are dropped.
usage: join_render.py <dir with wanted.jsonl> <task> <ntasks>   (writes <dir>/join_<task>.jsonl)"""
import json, os, sys, re, signal
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import pool_join as H
import dripper.process.simplify_html as sh
sh.tags_to_remove = {"head", "style", "script", "noscript", "link", "meta", "iframe", "frame"}; sh.ATTR_PATTERNS_TO_REMOVE = set()
from webpage_converter.convert import convert_html_to_structured_data
from bs4 import BeautifulSoup
def webkit_txt(html):
    raw = convert_html_to_structured_data(html, output_format="txt"); text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    text = re.sub(r"\|+", " ", text); text = re.sub(r"^[\s\-\|]+$", "", text, flags=re.M); text = re.sub(r"\\([\$\*\[\]_`#])", r"\1", text)
    return "\n".join(l for l in (re.sub(r"\s+", " ", x).strip() for x in text.split("\n")) if l)
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))
def render(h, sec=60):
    signal.alarm(sec)
    try: return webkit_txt(h)
    except Exception: return None
    finally: signal.alarm(0)
D = sys.argv[1]
tid, nt = int(sys.argv[2]), int(sys.argv[3])
groups = [json.loads(l) for l in open(D + "/wanted.jsonl", encoding="utf-8")][tid::nt]
out = open(f"{D}/join_{tid}.jsonl.tmp", "w", encoding="utf-8"); S = {"want": 0, "ok": 0, "no_candidate": 0, "text_mismatch": 0, "render_fail": 0, "shard_missing": 0}
for g in groups:
    stem, rows = g["stem"], g["rows"]; S["want"] += len(rows)
    s1_p = f"{H.ST1}/{stem}.jsonl"; s2_p = f"{H.ST2}/{stem}.jsonl.gz"
    if not (os.path.exists(s1_p) and os.path.exists(s2_p)): S["shard_missing"] += len(rows); continue
    s1 = []
    for l in open(s1_p, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            S["bad_json_line"] = S.get("bad_json_line", 0) + 1; continue   # tolerate a malformed step1 line
        mh = r.get("main_html") or ""
        if mh.strip(): s1.append((r.get("input") or "", mh, H.norm_head(mh)))
    s2 = [r["text"] for r in H.gz_rows(s2_p)]
    want = {r["row"]: r for r in rows}; i = 0
    for k, t in enumerate(s2):
        j = i; found = -1
        while j < len(s1) and j < i + 4:
            if H.head_in(t, s1[j][2]): found = j; break
            j += 1
        if found < 0: continue
        i = found + 1
        if k not in want: continue
        raw, mh, _ = s1[found]; w = want.pop(k)
        drip = render(mh)
        if drip is None: S["render_fail"] += 1; continue
        if drip != t: S["text_mismatch"] += 1; continue
        full = render(raw)
        if full is None or len(full) < 20: S["render_fail"] += 1; continue
        out.write(json.dumps({"i": w["i"], "stem": stem, "row": k, "tag": w["tag"], "input": full, "drip": t, "output": w["output"]}, ensure_ascii=False) + chr(10)); S["ok"] += 1
    S["no_candidate"] += len(want)
    if S["ok"] % 500 < len(rows): print(tid, S, flush=True)
out.close(); os.replace(f"{D}/join_{tid}.jsonl.tmp", f"{D}/join_{tid}.jsonl"); print("task", tid, "DONE", S, flush=True)
