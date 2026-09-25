"""Figure 1(b) numbers: how well do fixed rule stacks decide which pages belong in a pretraining corpus?
Compared against the independent gpt-oss-120b keep/remove judge (keep_judge.py; it never sees any pipeline's output),
on the same 5,000 held-out pages. Systems: RefinedWeb-rule and FineWeb-rule (rule stacks), UltraX and ProX-C (model
refiners), the teacher cascade, and ReScraper (release decoding). A page counts as kept when the system emits non-empty
text for it. Per system: kept, TP/FP/FN/TN against the judge, accuracy, precision, recall (= share of worth-keeping
pages kept) and lost_worthy; keep-drop accuracy of Figure 1(b) = mean of recall and the share of junk pages dropped
(TN / (TN + FP)). Also: per drop reason of each rule stack, pages dropped and how many of them the judge keeps.
usage: rule_motivation.py <heldout5k.jsonl> <heldout_pipeline work dir of the release run> <keep_judge_5000.jsonl> <out.json>"""
import json, sys, collections, math, os
PT, W, KJ, OUT = sys.argv[1:5]; W = W.rstrip("/") + "/"
G = {int(r["gid"]): r for r in map(json.loads, open(PT))}
O = {int(r["gid"]): r for r in map(json.loads, open(W + "ours.jsonl"))}
K = {int(r["gid"]): r["parsed"] for r in map(json.loads, open(KJ))}
jk = {g: K[g]["verdict"] == "keep" for g in G}
def load_keep(path, key="text"):
    d = {}
    for l in open(path):
        r = json.loads(l); t = r.get(key)
        d[int(r["gid"])] = bool(t and t != "None" and t.strip())
    return d
keep = {
    "RefinedWeb-rule": load_keep(W + "sys_refinedweb_rule.jsonl"),
    "FineWeb-rule": load_keep(W + "sys_fineweb_rule.jsonl"),
    "ProX-C": load_keep(W + "sys_proxc.jsonl"),
    "UltraX": load_keep(W + "sys_ultrax.jsonl"),
    "Teacher cascade": {g: G[g]["tag"] != "delete" for g in G},
    "ReScraper": {g: (O[g].get("status") == "ok" and O[g]["op"] != "delete") for g in G},
}
ORDER = ["RefinedWeb-rule", "FineWeb-rule", "ProX-C", "UltraX", "Teacher cascade", "ReScraper"]
R = {"n_pages": len(G), "judge_keep": sum(jk.values())}
def stats(f, pages):
    tp = sum(f[g] and jk[g] for g in pages); fp = sum(f[g] and not jk[g] for g in pages)
    fn = sum((not f[g]) and jk[g] for g in pages); tn = sum((not f[g]) and not jk[g] for g in pages)
    n = len(pages)
    return {"n": n, "kept": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "acc": (tp + tn) / n, "prec": tp / max(1, tp + fp), "rec": tp / max(1, tp + fn),
            "lost_worthy": fn / max(1, tp + fn)}
R["overall"] = {k: stats(keep[k], list(G)) for k in ORDER}
# which rule fires, and how often the judge disagrees
for name, path in (("RefinedWeb-rule", W + "sys_refinedweb_rule.jsonl"), ("FineWeb-rule", W + "sys_fineweb_rule.jsonl")):
    c = collections.defaultdict(lambda: [0, 0, 0])
    for l in open(path):
        r = json.loads(l); g = int(r["gid"])
        if keep[name][g]: continue
        k = (r.get("reason") or "unknown").split(":")[0]
        c[k][0] += 1; c[k][1] += jk[g]; c[k][2] += jk[g] and not keep["ReScraper"][g]
    R[name + "_drop_reasons"] = sorted(([k] + v for k, v in c.items()), key=lambda x: -x[1])
json.dump(R, open(OUT, "w"), indent=1, default=str)
print(f"{'system':>16} {'kept':>6} {'acc':>7} {'prec':>7} {'recall':>7} {'worthy pages lost':>18}")
for k in ORDER:
    v = R["overall"][k]
    print(f"{k:>16} {v['kept']:>6} {100*v['acc']:>6.1f}% {100*v['prec']:>6.1f}% {100*v['rec']:>6.1f}% {v['FN']:>8} ({100*v['lost_worthy']:.1f}%)")
for n in ("RefinedWeb-rule", "FineWeb-rule"):
    print("\n" + n, "drops (rule, pages dropped, of which judged worth keeping):")
    for k, a, b, o in R[n + "_drop_reasons"][:6]: print(f"   {k:>40} {a:>5} {b:>5} ({100*b/a:.0f}%) ours also drops {o:>4} ({100*o/a:.0f}%)")
