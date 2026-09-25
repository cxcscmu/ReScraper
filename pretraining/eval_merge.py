"""Merge chunked eval results (eval/shards/chunk0-6.yaml) into one metrics json, equivalent to a single full
run of eval/mmlu_and_lowvar.yaml.

Valid because get_aggregated_results() consumes ONLY data["eval_metrics"]["icl"] (a flat task->score dict): the
union of the chunks' icl dicts feeds the same aggregation code the monolithic run uses. Everything else in the
json (config echo, model name) is per-run noise.

Reads $DCLM_DIR/eval_results/chunks/<name>/<epoch>/metrics_chunk*.json and writes
$DCLM_DIR/eval_results/<name>/<epoch>/metrics_mmlu_and_lowvar.json. Refuses to write when tasks are missing, so a
partial merge can never masquerade as done. Completeness = suite parity: the merged icl must reproduce exactly the
task set of mmlu_and_lowvar.yaml (the json's own "missing tasks" field lists tasks absent from the whole DCLM eval
universe and is never empty for this suite). The Core score reported in the paper is then computed from the merged
json by eval_sheet.py (core_sheet.sh), never read from the json.
usage: eval_merge.py <name>
"""
import json, glob, os, re, sys
sys.path.insert(0, os.path.join(os.environ["DCLM_DIR"], "eval"))
import pandas as pd
from aggregated_metrics import get_aggregated_results

C = os.environ["DCLM_DIR"]
EXPECTED = 7


def suite_tasks():
    """The authoritative task set: whatever the monolithic mmlu_and_lowvar.yaml runs."""
    return set(re.findall(r"label:\s*(\S+)", open(f"{C}/eval/mmlu_and_lowvar.yaml").read()))


def merge(variant):
    d = glob.glob(f"{C}/eval_results/chunks/{variant}/epoch_*")
    if not d:
        return f"{variant}: no chunk dir yet"
    d = d[0]
    epoch = os.path.basename(d)
    files = sorted(glob.glob(f"{d}/metrics_chunk*.json"))
    if len(files) < EXPECTED:
        return f"{variant}: {len(files)}/{EXPECTED} chunks"
    icl, base = {}, None
    for f in files:
        j = json.load(open(f))
        got = j.get("eval_metrics", {}).get("icl", {})
        if not got:
            return f"{variant}: {f} has empty icl -- not merging"
        dup = set(icl) & set(got)
        if dup:
            return f"{variant}: duplicate tasks across chunks {dup} -- refusing"
        icl.update(got)
        base = base or j

    # Completeness: exact parity with the monolithic suite.
    suite = suite_tasks()
    miss, extra = sorted(suite - set(icl)), sorted(set(icl) - suite)
    if miss or extra:
        return f"{variant}: SUITE MISMATCH missing={miss} extra={extra} -- not writing"

    base["eval_metrics"]["icl"] = icl
    meta = pd.read_csv(f"{C}/eval/eval_meta_data.csv")
    agg = json.load(open(f"{C}/eval/additional_aggregation.json"))
    out = get_aggregated_results(base, meta, agg)
    if not isinstance(out.get("Core"), float):
        return f"{variant}: Core not numeric ({out.get('Core')!r}) -- not writing"

    path = f"{C}/eval_results/{variant}/{epoch}/metrics_mmlu_and_lowvar.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"), indent=4)
    os.replace(tmp, path)
    return f"{variant}: MERGED_OK {len(icl)} tasks Core={out.get('Core')} -> {path}"


import sys
for v in (sys.argv[1],):
    print(merge(v))
