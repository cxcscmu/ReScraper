#!/usr/bin/env python3
"""Write the hand-picked held-out pages of the project page into assets/cases.js."""
import json, re, os

OUT = os.path.expanduser("~/rescraper-site/assets/cases.js")
POOL = "pool.jsonl"

KEEP = [600, 173, 2403, 3544, 192, 503, 937, 1302, 1791, 2343, 3149, 4500]
EDIT = [11, 758, 4645, 4762, 2586, 2367, 2556, 3498, 1974, 100, 4334, 946]
DELETE = [220, 237, 1802, 1787, 3198, 1778, 3480, 1222]
REWRITE = [1417, 4655, 4410, 4609, 2614, 1690, 3243]

MAXSRC = 16000
MAXOUT = 9000
rows = {json.loads(l)["gid"]: json.loads(l) for l in open(POOL, encoding="utf-8")}


def clip(t, n):
    t = t or ""
    if len(t) <= n:
        return {"t": t, "cut": False}
    cut = t[:n]
    cut = cut[:cut.rfind("\n")] if "\n" in cut[n // 2:] else cut
    return {"t": cut, "cut": True}


def words(t):
    return len((t or "").split())


def domain(u):
    m = re.match(r"https?://([^/]+)", u or "")
    return (m.group(1) if m else "").lower().removeprefix("www.")


# round-robin so browsing in order shows all four operations
order, pools = [], [("keep", list(KEEP)), ("edit", list(EDIT)),
                    ("rewrite", list(REWRITE)), ("delete", list(DELETE))]
while any(p for _, p in pools):
    for _, p in pools:
        if p:
            order.append(p.pop(0))

SYS = [("refinedweb_rule", "RefinedWeb-rule"), ("fineweb_rule", "FineWeb-rule"),
       ("proxc", "ProX-C"), ("ultrax", "UltraX")]
cases = []
for i, gid in enumerate(order):
    r = rows[gid]
    post = r["post"] or {}
    out = []
    for key, label in SYS:
        s = r[key]
        sc = post.get(key) or {}
        out.append({"key": key, "label": label, "status": s["status"],
                    "words": words(s["text"]), "dataman": sc.get("dataman"),
                    "edu": sc.get("fineweb_edu"), "note": s.get("program") or s.get("reason"),
                    **clip(s["text"], MAXOUT)})
    # the raw program of a <rewrite> ends with the whole rewritten page: keep only the ops
    prog = (r["ours_raw"] or "").rstrip()
    if r["ours_op"] == "rewrite":
        lines = prog.split("\n")
        tags = [k for k, l in enumerate(lines) if re.fullmatch(r"<(keep|edit|delete|rewrite)>", l.strip())]
        if len(tags) > 1:
            prog = "\n".join(lines[:tags[-1] + 1]) + "\n… (the rewritten page is shown below)"
    if len(prog) > 900:
        prog = prog[:900].rsplit("\n", 1)[0] + "\n…"

    sc = post.get("rescraper") or {}
    out.append({"key": "rescraper", "label": "ReScraper", "status": r["ours_op"],
                "words": words(r["ours_text"]), "dataman": sc.get("dataman"),
                "edu": sc.get("fineweb_edu"), "program": prog,
                **clip(r["ours_text"], MAXOUT)})
    f = r["faith"] or {}
    cases.append({
        "i": i, "gid": gid, "url": r["url"], "domain": domain(r["url"]),
        "op": r["ours_op"],
        "judge": {"verdict": r["judge"]["verdict"], "value": r["judge"]["value"],
                  "reason": r["judge"]["reason"]},
        "pre": {"dataman": (r["pre"] or {}).get("dataman"),
                "edu": (r["pre"] or {}).get("fineweb_edu")},
        "source": clip(r["rendered"], MAXSRC),
        "resiliparse": clip(r["resiliparse"], MAXOUT),
        "faith": ({"bs_F": round(f["bs_F"], 3), "src_words": f["src_words"],
                   "rw_words": f["rw_words"]} if f and r["ours_op"] == "rewrite" else None),
        "out": out,
    })

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("window.CASES = ")
    json.dump(cases, fh, ensure_ascii=True, separators=(",", ":"))
    fh.write(";\n")
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB,", len(cases), "cases")
print("ops:", {k: sum(c["op"] == k for c in cases) for k in ("keep", "edit", "delete", "rewrite")})
