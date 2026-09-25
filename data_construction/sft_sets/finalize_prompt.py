"""Fill the __FREQS__ placeholder of the student system prompt (prompts/student_system.tmpl) from the SFT file
it will actually be trained on. Stage 1 -> prompts/student_system_stage1.txt, Stage 2 -> prompts/student_system_stage2.txt.

The prompt quotes the outcome frequencies; if the rescue flips change them the quoted numbers become
a lie about the data, so they are measured rather than typed in. Rare tags are printed with a decimal:
the rescue bands are well under 1%, and rounding <rewrite> to "0%" would tell the student the tag
never occurs, which is both false and the opposite of what it is being taught.
usage: finalize_prompt.py <tmpl> <sft.jsonl> <out.txt>
"""
import json, sys, collections
tmpl, src, dst = sys.argv[1:4]
c = collections.Counter(); n = 0
with open(src, encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)["output"]
        c[t.split("\n", 1)[0].strip()] += 1; n += 1
order = ["<keep>", "<edit>", "<delete>", "<rewrite>"]
assert set(c) <= set(order), ("unexpected tags", [k for k in c if k not in order])


def pct(k):
    p = 100.0 * c[k] / n
    return "%s %d%%" % (k, round(p)) if p >= 1 else "%s %.1f%%" % (k, p)


freqs = ", ".join(pct(t) for t in order if c[t])
text = open(tmpl, encoding="utf-8").read()
assert "__FREQS__" in text, "placeholder already filled"
open(dst, "w", encoding="utf-8").write(text.replace("__FREQS__", freqs + "."))
print("rows %d | %s" % (n, dict(c)), flush=True)
print("WROTE", dst, "->", freqs)
