"""Line -> DOM origin: the HTML block element each rendered line came from.

Used by the Stage-1 rescue rule (data_construction/rescue/rescue_build.py) to find prose blocks. The element label
carries real signal: measured 0.59 bits of information gain on keep/drop (p 0.78 keep, div 0.13, li 0.00, h2 0.94).

annotate_lines(raw_html, lines) -> list of labels ("" when the line cannot be attributed).
Attribution is by containment in a block element's own text - it never perturbs the rendering (injecting
sentinels does: it cut a 3,128-line render to 1,070). Two passes: exact containment in the smallest block,
then, for lines still unattributed, the block whose text has the largest word overlap with the line
(handles table rows and blocks that render to several lines).
"""
import re
from bs4 import BeautifulSoup
BLOCKY = {"p", "div", "td", "th", "li", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "dt", "dd",
          "figcaption", "caption", "tr", "table", "ul", "ol", "nav", "footer", "header", "aside", "article",
          "section", "form", "label", "button"}
WS = re.compile(r"\s+"); CLS = re.compile(r"[^a-z0-9]+")
def _label(el):
    cls = " ".join((el.get("class") or [])[:2]) or (el.get("id") or "")
    cls = CLS.sub(".", cls.lower()).strip(".")
    return (el.name + ("." + cls if cls else ""))[:24]
def annotate_lines(raw_html, lines, max_elems=4000):
    try: soup = BeautifulSoup(raw_html, "html.parser")
    except Exception: return [""] * len(lines)
    els = []
    for el in soup.find_all(True):
        if el.name not in BLOCKY: continue
        txt = WS.sub(" ", el.get_text(separator=" ")).strip().lower()
        if not txt or len(txt) > 6000: continue
        els.append((len(txt), txt, _label(el)))
        if len(els) >= max_elems: break
    els.sort()
    out = []
    for ln in lines:
        low = WS.sub(" ", ln).strip().lower(); lab = ""
        if low:
            for L, t, name in els:
                if L >= len(low) and low in t: lab = name; break
            if not lab:                                   # fall back to best word overlap
                w = set(low.split()); best = 0.0
                if w:
                    for L, t, name in els:
                        if L < 12: continue
                        tw = set(t.split()); ov = len(w & tw) / len(w)
                        if ov > best: best, lab = ov, name
                    if best < 0.6: lab = ""
        out.append(lab)
    return out
def number_lines_dom(text, raw_html):
    """'<lid:N|tag.class> line' (or '<lid:N> line' when unattributed) - the annotated input."""
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    labs = annotate_lines(raw_html, lines)
    return "\n".join("<lid:%d%s> %s" % (i + 1, ("|" + labs[i]) if labs[i] else "", lines[i]) for i in range(len(lines)))
