"""Strip repro-rephraser thinking traces and meta-preambles from a rewrite.

repro-rephraser-4B is Qwen3-based and emits <think> blocks; the original extraction took everything
after the FIRST "Here is a paraphrased version:", which inside a think block captured the tail of the
reasoning. Take the text after the LAST </think> and after the LAST marker instead.
"""
import re

PARA = re.compile(r"here\s+is\s+a\s+paraphrased\s+version\s*:?\s*", re.I)
THINK_OPEN = re.compile(r"<think>", re.I)
DIRTY = re.compile(r"</?think>|the user wants|i need to (make sure|rephrase|paraphrase)|here is a paraphrased version", re.I)


def clean(t):
    if not t:
        return ""
    low = t.lower()
    i = low.rfind("</think>")
    if i >= 0:
        t = t[i + len("</think>"):]
    m = None
    for m in PARA.finditer(t):
        pass
    if m:
        t = t[m.end():]
    t = THINK_OPEN.sub("", t)
    t = re.sub(r"^\s*(-{3,}|`{3,}[a-zA-Z]*)\s*", "", t)
    t = re.sub(r"\s*(`{3,})\s*$", "", t)
    lines = [" ".join(x.split()) for x in t.split("\n")]
    return "\n".join(x for x in lines if x).strip()


def is_dirty(t):
    return bool(DIRTY.search(t or ""))
