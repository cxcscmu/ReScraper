"""FineWeb-rule stack (datatrove 0.2.0, the FineWeb-release version), as built by the FineWeb-rule baseline
(baselines/): filters in the order of datatrove's examples/fineweb.py, with its arguments:
    URLFilter()  -> [extraction]  -> LanguageFilter(en, 0.65) -> GopherRepetitionFilter() -> GopherQualityFilter()
    -> C4QualityFilter(filter_no_terminal_punct=False) (line-level edits applied) -> FineWebQualityFilter()
LanguageFilter uses fastText lid.176.bin (the same file datatrove downloads; the DCLM copy, lib/render.LID_PATH).
No MinHash dedup, no PII formatter. Callers remove blank/whitespace-only lines before the filters (resiliparse separates
blocks with blank lines, trafilatura - FineWeb's extractor - does not; Gopher's line rules would count every "" line).
Used by stages/s2_fw_rule.py, judges/allrules/fw_allrules.py and diversity/fw_scale.py (DATATROVE_PY env).
"""
import os, sys
from importlib.metadata import version

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "lib"))
from render import LID_PATH as LID  # noqa: E402

assert version("datatrove") == "0.2.0", "FineWeb-rule needs datatrove==0.2.0, found %s" % version("datatrove")
from datatrove.data import Document  # noqa: E402,F401
from datatrove.pipeline.filters import (URLFilter, LanguageFilter, GopherRepetitionFilter, GopherQualityFilter,  # noqa: E402
                                        C4QualityFilter, FineWebQualityFilter)


def build():
    from fasttext.FastText import _FastText
    lf = LanguageFilter(); lf._model = _FastText(LID)
    uf = URLFilter(); uf.download_data()
    return uf, [("lang", lf), ("gopher_rep", GopherRepetitionFilter()), ("gopher_qual", GopherQualityFilter()),
                ("c4", C4QualityFilter(filter_no_terminal_punct=False)), ("fineweb", FineWebQualityFilter(exclusion_writer=None))]


def res(r):
    if isinstance(r, tuple):
        return bool(r[0]), r[1]
    return bool(r), None
