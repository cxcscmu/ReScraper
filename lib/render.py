"""Canonical page rendering used for every ReScraper input (SFT, held-out and pool inference), plus the text
post-processing and filters applied to Dripper's extracted text when sampling SFT seed pages.

webkit_txt(html): Dripper's (MinerU-HTML 1.0.0) WebKit-style text renderer with head/style/script/noscript/
    link/meta/iframe/frame removed and attribute-pattern removal disabled, then a BeautifulSoup markup strip,
    `|` collapse, separator-line removal, Markdown unescape, whitespace collapse and empty-line removal.
    The model input is `editops.number_lines(webkit_txt(html))`.
webkit_post(text): the same trailing cleanup, applied to an already rendered text (e.g. Dripper's step-2 text).
MARKUP: residual-markup detector used to drop seed pages whose extracted text leaks HTML.
LID_PATH: fastText lid.176 language-id model (the copy shipped with DCLM); override with LID_MODEL.
"""
import os, re

LID_PATH = os.environ.get("LID_MODEL", os.path.join(os.environ.get("DCLM_DIR", ""),
                          "baselines/mappers/enrichers/language_id_enrichment_models/lid.176.bin"))

MARKUP = re.compile(r"</\w+>|<\\/\w+>|<(div|span|nav|ul|li|table|tr|td)\b[^>]*>", re.I)


def webkit_post(t):
    """Trailing cleanup applied after rendering."""
    t = re.sub(r"\|+", " ", t)
    t = re.sub(r"^[\s\-\|]+$", "", t, flags=re.MULTILINE)
    t = re.sub(r"\\([\$\*\[\]_`#])", r"\1", t)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in t.split("\n")]
    return "\n".join(l for l in lines if l)


_converter = None


def _load_converter():
    global _converter
    if _converter is None:
        import dripper.process.simplify_html as sh
        sh.tags_to_remove = {"head", "style", "script", "noscript", "link", "meta", "iframe", "frame"}
        sh.ATTR_PATTERNS_TO_REMOVE = set()
        from webpage_converter.convert import convert_html_to_structured_data
        _converter = convert_html_to_structured_data
    return _converter


def webkit_txt(html):
    from bs4 import BeautifulSoup
    raw = _load_converter()(html, output_format="txt")
    t = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    t = re.sub(r"\|+", " ", t); t = re.sub(r"^[\s\-\|]+$", "", t, flags=re.M); t = re.sub(r"\\([\$\*\[\]_`#])", r"\1", t)
    return "\n".join(l for l in (re.sub(r"\s+", " ", x).strip() for x in t.split("\n")) if l)
