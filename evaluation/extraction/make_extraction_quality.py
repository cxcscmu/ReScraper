"""Assemble the figure data of the extraction figure (Figure 8) from one run of run_all.sh:
{"pages", "source", "ext_vs_dripper": <ext_vs_dripper_<N>.json>, "judge": <judge/ext_judge_<N>.json>}
(= extraction_quality.json in analysis/data).
usage: make_extraction_quality.py <run_dir> <N> <out.json> [source description]"""
import json, sys
D, N, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SRC = sys.argv[4] if len(sys.argv) > 4 else D
R = {"pages": N, "source": SRC,
     "ext_vs_dripper": json.load(open("%s/ext_vs_dripper_%d.json" % (D, N))),
     "judge": json.load(open("%s/judge/ext_judge_%d.json" % (D, N)))}
json.dump(R, open(OUT, "w"), indent=1)
print("wrote", OUT)
