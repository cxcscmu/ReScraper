"""RePro rephraser (1B, OLMo-2 based) conversation format and output parsing, shared by every script that
generates <rewrite> targets (data_construction/rescue/*).

Follows RePro upstream rl/src/infer/run_infer.py line for line: same recycle_prompt, same system
prompt, same SamplingParams, the same 7000-char chunk / rejoin-with-space handling of long pages,
the same first-match extraction, and the same leading/trailing "---" strip. Upstream sets
max_prompt_tokens = 8192 only for Qwen (where it also appends " /no_think"); OLMo2 has no thinking
mode, so this takes the 4096 branch with the system prompt unchanged.

The rephraser checkpoint is the public RePro 1B rephraser (identifier withheld for anonymity); pass it
via the RW_MODEL environment variable. The prompt text is also stored in prompts/repro_*.txt.
"""
import re
from vllm import LLM, SamplingParams

# --- verbatim from RePro rl/src/infer/run_infer.py ---
recycle_prompt = """Your task is to read and paraphrase the provided text following these instructions:
- Delete clearly irrelevant content:
  - Website headers, navigation bars, or menu items (e.g., "Home | About | Contact")
  - Unrelated HTTP links (e.g., ads, trackers, developer tools)
  - Generic footers (e.g., contact info, privacy policies, unsubscribe links)
  - Empty lines or decorative elements (e.g., "---")
- Preserve all content that is relevant and meaningful:
  - Informative or independently useful
  - Related to the topic, even tangentially
  - Provides context, background, or supporting value
  - Includes technical terms, key concepts, factual details, reasoning, and examples
- Handle mixed-relevance sentences carefully:
  - Remove only the irrelevant fragment if the rest remains coherent
  - Delete the whole sentence if the remainder loses meaning
- Do not alter meaningful content unnecessarily:
  - Only delete or modify when content is clearly meaningless or off-topic
  - Preserve the original structure, logic, and depth of the text
- Do not add explanations, notes, assumptions, or claims not found in the original text
Here is the text:
{TEXT}
Task:
After thoroughly reading the above text, paraphrase it in high-quality and clear English following the instructions.
Start your response immediately with "Here is a paraphrased version:" and then provide the paraphrased text."""

SYSTEM = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions."
sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=2048)
MAX_PROMPT_TOKENS = 4096   # upstream: 8192 only when "Qwen" in model
CHUNK = 7000               # upstream splits pages into 7000-char chunks and rejoins with " "


def make_conversation(text, tokenizer, max_prompt_tokens=MAX_PROMPT_TOKENS):
    base_tokens = tokenizer.encode(recycle_prompt.replace("{TEXT}", ""))
    system_tokens = tokenizer.encode(SYSTEM)
    available = max_prompt_tokens - len(base_tokens) - len(system_tokens) - 50
    ids = tokenizer.encode(text)
    if len(ids) > available:
        text = tokenizer.decode(ids[: available // 2]) + "... " + tokenizer.decode(ids[-(available // 2):])
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": recycle_prompt.format(TEXT=text)}]


def extract(content):
    m = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def strip_rules(t):
    if t[:3] == "---":
        t = t[3:]
    if t[-3:] == "---":
        t = t[:-3]
    return t.strip()
