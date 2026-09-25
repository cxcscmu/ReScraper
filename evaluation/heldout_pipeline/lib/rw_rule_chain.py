"""RefinedWeb-rule stack of DCLM, applied page by page (used by stages/s2_rw_rule.py and judges/allrules/rw_allrules.py).

DCLM's own mapper factory baselines.core.factories.get_mapper(_safe=True), applied page by page exactly as DCLM's
process_single_file does (an exception drops the page), with the nltk punkt_tab patch. Config
dclm_baseline_refinedweb_post_lang.yaml = the rule stack of the resiliparse RefinedWeb-rule baseline
(resiliparse_extract -> post_lang -> bff dedup). Needs the DCLM checkout ($DCLM_DIR) and the DCLM env (DCLM_PY).
Importing this module changes the working directory to $DCLM_DIR (the mappers resolve their asset files relative to it).
"""
import os, sys, yaml

DCLM = os.environ.get("DCLM_DIR") or sys.exit("set DCLM_DIR (DCLM checkout)")
os.chdir(DCLM); sys.path.insert(0, DCLM)
from baselines.core.factories import get_mapper  # noqa: E402
import baselines.mappers.core_utils as _cu  # noqa: E402
from nltk.tokenize.punkt import PunktTokenizer  # noqa: E402
_cu.sent_tokenizer = PunktTokenizer("english")
CFG = os.path.join(DCLM, "baselines", "baselines_configs") + "/"
TLDS = os.path.join(DCLM, "baselines", "mappers", "iana_tlds.txt")


def steps_of(name, drop=()):
    st = yaml.safe_load(open(CFG + name))[0]["steps"]
    out = []
    for s in st:
        s = dict(s); s.pop("_aggregate", None)
        if s["func"] in drop: continue
        if s.get("banlist_from_fname") and not os.path.exists(os.path.join(DCLM, s["banlist_from_fname"])):
            print("SKIP step (banlist file not shipped with the DCLM repo):", name, s["func"], s["banlist_from_fname"]); continue
        if s["func"] == "url_removal_modifier": s.setdefault("tlds_filepath", TLDS)
        out.append(s)
    return out


def build(steps):
    return [(s["func"], get_mapper(**s, _safe=True)) for s in steps]


def run(stack, page):
    pages = [page]
    for name, f in stack:
        nxt = []
        for p in pages:
            r = f(p)
            if isinstance(r, list): nxt.extend(r)
            else: return [], name + " ERROR " + str(r)[:200]
        pages = nxt
        if not pages: return [], name
    return pages, None
