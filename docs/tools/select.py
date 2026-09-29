#!/usr/bin/env python3
"""Rank candidate cases per ReScraper operation for the project-page viewer."""
import json, sys, re, collections

POOL = "pool.jsonl"
rows = [json.loads(l) for l in open(POOL, encoding="utf-8")]


def w(t):
    return len((t or "").split())


def dm(rec, k):
    p = (rec.get("post") or {}).get(k)
    return None if not p else p.get("dataman")


def domain(u):
    m = re.match(r"https?://([^/]+)", u or "")
    return (m.group(1) if m else "?").lower().removeprefix("www.")


def ascii_ok(t):
    t = t or ""
    if not t:
        return False
    nonascii = sum(1 for c in t if ord(c) > 127)
    return nonascii / max(1, len(t)) < 0.02


def dropped(rec, k):
    return (rec[k].get("text") or "").strip() == ""


cand = collections.defaultdict(list)
for r in rows:
    if r["ours_status"] != "ok" or not r["ours_op"]:
        continue
    rend = r["rendered"] or ""
    if not (400 < len(rend) < 14000):
        continue
    if not ascii_ok(rend):
        continue
    ot = (r["ours_text"] or "")
    op = r["ours_op"]
    j = r["judge"] or {}
    ndrop = sum(dropped(r, k) for k in ("ultrax", "proxc", "refinedweb_rule", "fineweb_rule"))
    score = 0.0
    if op in ("keep", "edit"):
        if not (150 < len(ot) < 7000):
            continue
        if j.get("verdict") != "keep":
            continue
        score += 2.0 * j.get("value", 0)
        score += 2.0 * ndrop                      # baselines threw away a page worth keeping
        # UltraX/ProX keep boilerplate that ours removes
        for k in ("ultrax", "proxc"):
            t = r[k].get("text") or ""
            if t and w(t) > w(ot) * 1.35:
                score += 1.5
        d_ours, d_ux = dm(r, "rescraper"), dm(r, "ultrax")
        if d_ours and d_ux and d_ours > d_ux:
            score += 1.0
        if op == "edit":
            score += 1.0
    elif op == "delete":
        if j.get("verdict") != "remove":
            continue
        score += (3 - j.get("value", 3)) * 2.0
        kept = [k for k in ("ultrax", "proxc", "refinedweb_rule", "fineweb_rule") if not dropped(r, k)]
        if not kept:
            continue
        score += 2.5 * len(kept)                  # junk the baselines let through
    elif op == "rewrite":
        f = r["faith"] or {}
        if not f:
            continue
        if f.get("ent_novel_page_n", 9) != 0 or f.get("num_novel_page_n", 9) != 0:
            continue                              # any entity/number not in the page -> drop
        if f.get("bs_F", 0) < 0.92:
            continue
        if not (150 < len(ot) < 6000):
            continue
        if f.get("rw_words", 0) < 60:
            continue
        lines = [l for l in ot.split("\n") if l.strip()]
        if len(lines) < 2 or f["rw_words"] / len(lines) < 12:
            continue                              # prose, not a list of titles
        if j.get("verdict") != "keep" or j.get("value", 0) < 2:
            continue                              # only rescue pages a judge calls worth keeping
        score += 6 * (f["bs_F"] - 0.92) * 10
        score += 1.5 * ndrop
        post = (r["post"] or {}).get("rescraper") or {}
        score += 2.0 * min(2.0, post.get("fineweb_edu") or 0)
        d_ours = dm(r, "rescraper")
        if d_ours and d_ours >= 4:
            score += 1.5
    cand[op].append((score, r))

for op in ("keep", "edit", "delete", "rewrite"):
    cand[op].sort(key=lambda x: -x[0])
    print(f"== {op}: {len(cand[op])} candidates")

json.dump({op: [r["gid"] for _, r in cand[op][:60]] for op in cand}, open("cand_gids.json", "w"))

if len(sys.argv) > 1 and sys.argv[1] == "show":
    op = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    for s, r in cand[op][off:off + n]:
        print("=" * 100)
        print(f"gid {r['gid']}  score {s:.1f}  {domain(r['url'])}  {r['url']}")
        print(f"  judge {r['judge']}  faith {r['faith']}")
        print(f"  dm: ours={dm(r,'rescraper')} ux={dm(r,'ultrax')} prox={dm(r,'proxc')} rw={dm(r,'refinedweb_rule')} fw={dm(r,'fineweb_rule')}")
        print(f"  ours program: {(r['ours_raw'] or '')[:200]}")
        for k in ("ultrax", "proxc", "refinedweb_rule", "fineweb_rule"):
            t = r[k].get("text")
            print(f"  {k}: {r[k]['status']} words={w(t)}")
        print("  --- RENDERED (first 700) ---")
        print("   " + (r["rendered"] or "")[:700].replace("\n", "\n   "))
        print("  --- OURS ---")
        print("   " + (r["ours_text"] or "")[:2500].replace("\n", "\n   "))
