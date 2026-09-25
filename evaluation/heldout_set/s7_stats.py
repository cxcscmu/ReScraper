"""Summary tables of the held-out page table (label mix per subset and shard, relabel transitions, paraphrase and edu
sources, edu distribution, resiliparse join). usage: s7_stats.py <heldout5k.jsonl>"""
import sys, json, collections, statistics
R = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
T4 = ("keep", "edit", "delete", "rewrite")
def mix(rs):
    c = collections.Counter(r["tag"] for r in rs); n = len(rs)
    return "| %d | %s |" % (n, " | ".join("%d (%.1f%%)" % (c[t], 100.0 * c[t] / max(1, n)) for t in T4))
print("## label mix\n| subset | rows | keep | edit | delete | rewrite |\n|---|---|---|---|---|---|")
for name, rs in (("all 5,000", R), ("rows 0-959 (heldout960)", [r for r in R if r["source"] == "heldout960"]),
                 ("rows 960-3160 (relabelled)", [r for r in R if r["source"] == "relabel2201"]),
                 ("rows 3161-4999 (new shards)", [r for r in R if r["source"] == "new_shard"])):
    print("| %s %s" % (name, mix(rs)))
for s in sorted(set(r["stem"] for r in R)):
    print("| stem %s %s" % (s, mix([r for r in R if r["stem"] == s])))
print("\n## old -> new tag (rows 0..3160)")
for src in ("heldout960", "relabel2201"):
    c = collections.Counter((r["old_tag"], r["tag"]) for r in R if r["source"] == src)
    print(src, {"%s->%s" % k: v for k, v in sorted(c.items())})
print("\n## paraphrase sources (rewrite rows)")
for src in ("heldout960", "relabel2201", "new_shard"):
    c = collections.Counter((r["paraphrase_source"] or "-").split(" ")[0] for r in R if r["source"] == src and r["tag"] == "rewrite")
    print(src, dict(c))
print("\n## teacher-deleted pages with edu>=1.0 but no passing paraphrase (stay delete)")
for src in ("heldout960", "relabel2201", "new_shard"):
    rs = [r for r in R if r["source"] == src and r["tag"] == "delete" and r["edu"] is not None and r["edu"] >= 1.0]
    print(src, len(rs), collections.Counter((r["paraphrase_source"] or "-") for r in rs))
print("\n## edu sources"); print(dict(collections.Counter((r["source"], r["edu_source"].split("(")[0]) for r in R)))
print("\n## edu distribution (value used by the rule; recomputed for keep/edit pages without a text_decisions entry)")
bins = [(-9, 0), (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 9)]
def hist(rs):
    e = [r["edu"] for r in rs]
    return " | ".join("%d" % sum(1 for x in e if a <= x < b) for a, b in bins), statistics.median(e) if e else None, statistics.mean(e) if e else None
print("| subset | n | <0 | 0-0.5 | 0.5-1.0 | 1.0-1.5 | 1.5-2.0 | >=2.0 | median | mean |\n|---|---|---|---|---|---|---|---|---|---|")
for name, rs in (("all", R), ("teacher-deleted (delete+rewrite)", [r for r in R if r["tag"] in ("delete", "rewrite")]),
                 ("keep", [r for r in R if r["tag"] == "keep"]), ("edit", [r for r in R if r["tag"] == "edit"])):
    h, md, mn = hist(rs); print("| %s | %d | %s | %.3f | %.3f |" % (name, len(rs), h, md, mn))
print("\n## resiliparse join")
print(dict(collections.Counter(r["resiliparse_join"] for r in R)))
print({"%s/%s" % k: v for k, v in sorted(collections.Counter((r["source"], r["resiliparse_join"]) for r in R).items())})
print("empty resiliparse text among joined:", sum(1 for r in R if r["resiliparse"] is not None and not r["resiliparse"].strip()))
print("\n## new-shard staged kinds / 27B finish", dict(collections.Counter(r.get("staged_kind") for r in R if r["source"] == "new_shard")),
      dict(collections.Counter(r.get("t27_finish") for r in R if r["source"] == "new_shard")))
print("new-shard 27B tag", dict(collections.Counter(r.get("t27_tag") for r in R if r["source"] == "new_shard")))
print("pool_student_kept (two-stage refiner) vs tag:", {"%s/%s" % k: v for k, v in sorted(collections.Counter((r["tag"], str(r["pool_student_kept"])) for r in R).items())})
print("gid check:", all(r["gid"] == k for k, r in enumerate(R)), "rows", len(R))
