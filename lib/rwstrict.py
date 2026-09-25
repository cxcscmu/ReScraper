"""One definition of "this text is the rephraser talking about its task", shared by every consumer.

Why this exists: rwclean.is_dirty() matches only <think>, "the user wants", "i need to
rephrase/paraphrase/make sure" and "here is a paraphrased version". Measured over 1,219,606
documents of an earlier rewritten corpus, that set covers 485 of 4,702 contaminated documents; on
another rebuild it dropped 28 rows out of 150,557 rewrites. The other ~90% are shapes it was
never written for:

    self_commentary   1665   "This paraphrased version retains the original details..."
    instr_compliance  1450   "...while ensuring clarity", "retains all relevant details"
    narration_strict  1286   a line that opens "Okay, the user" / "Let me rephrase"
    prompt_echo        312   the recycling instructions themselves, verbatim
    answer_preamble    485   "Here is a paraphrased version:" left inline
    think_tag            0   gone since the switch to the non-thinking rephraser

Any corpus audit should use the same patterns: the filter and the audit must agree, or the
audit keeps passing data the filter let through. Patterns deliberately EXCLUDED because they fire
on ordinary web prose (checked against real hits):
    "so the user could change the length ..."      <- bare "the user"
    "Let me start by saying I can't recommend ..." <- bare "let me start by"
"""
import re

PATTERNS = {
    "think_tag":        re.compile(r"</?think>", re.I),
    "prompt_echo":      re.compile(r"your task is to read and paraphrase|delete clearly irrelevant content", re.I),
    "answer_preamble":  re.compile(r"here is a paraphrased version", re.I),
    "self_commentary":  re.compile(r"\bthis (paraphrased version|paraphrase) (retains|maintains|keeps|preserves)", re.I),
    "instr_compliance": re.compile(r"\b(retains all relevant details|while ensuring clarity|adherence to the instructions)\b", re.I),
    "narration_strict": re.compile(r"(?m)^\s*(?:(?:okay|alright)[,.]? the user\b"
                                   r"|let me (?:reorganize|rephrase|paraphrase|rewrite) )", re.I),
    "needs_to":         re.compile(r"\b(the user wants|i need to (?:make sure|rephrase|paraphrase))\b", re.I),
}


def is_dirty(t):
    if not t:
        return False
    return any(rx.search(t) for rx in PATTERNS.values())


def which(t):
    return [n for n, rx in PATTERNS.items() if t and rx.search(t)]
