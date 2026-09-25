"""Light prepack check (minutes, not an hour): a full check walks every bin in Python, which on a
1M prepack (~115k bins x 32k tokens x 3 columns) takes 30-60 min on the critical path. This checks the
structure and the first/last 100 bins with the same assertions, plus stats.json consistency.
usage: prepack_light_check.py <prepack_dir>"""
import sys, json, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import sft_train as S
from datasets import load_from_disk
d = sys.argv[1]; st = json.load(open(d + "/stats.json")); ds = load_from_disk(d); tr, va = ds["train"], ds["validation"]
assert len(tr) == st["bins"], (len(tr), st["bins"])
idx = list(range(min(100, len(tr)))) + list(range(max(0, len(tr) - 100), len(tr)))
tot = restarts = 0; mx = 0
for i in idx:
    ex = tr[i]; n = len(ex["input_ids"]); assert 0 < n <= st["max_seq_len"] and n == len(ex["labels"]) == len(ex["position_ids"]), i
    tot += n; mx = max(mx, n); restarts += sum(1 for p in ex["position_ids"] if p == 0)
    assert any(l != -100 for l in ex["labels"]), "bin %d has no supervised token" % i
col = S.PackedDataCollator(pad_token_id=0, pad_to_multiple_of=8); b = col([tr[0], tr[1]])
assert b["input_ids"].dtype == torch.int64 and "position_ids" in b and "attention_mask" not in b, b.keys()
bv = col([va[0], va[1]]); assert "attention_mask" in bv and "position_ids" not in bv
raw = json.load(open(d + "/val_raw.json", encoding="utf-8")); assert len(raw) == len(va), (len(raw), len(va))
print("LIGHT_CHECK_OK bins", len(tr), "checked", len(idx), "tokens_in_checked", tot, "samples_in_checked", restarts, "max bin", mx, "val", len(va), "stats tokens", st["tokens"], "samples", st["samples"])
