#!/usr/bin/env bash
# Export an HTML/SVG figure to a tightly cropped PDF (Figure 2, fig:method).
#
#   analysis/export_html_figure.sh [input.html] [output.pdf] [margin_pt]
#
# Defaults: $RESCRAPER_ROOT/analysis/figure_sources/method_overview.html -> $FIG_DIR/method.pdf
# (FIG_DIR defaults to $RESCRAPER_ROOT/analysis/figures), 1pt margin.
#
# Pipeline: headless Chrome/Chromium prints the page to PDF (the page's @page rule zeroes the
# margins, which also suppresses Chrome's date/title/URL header and footer), then
# Ghostscript measures the ink bounding box and re-pages the PDF to that box plus the
# margin. Requires Chrome or Chromium (path in $CHROME, else the first of google-chrome,
# chromium, chromium-browser on PATH) and gs (Ghostscript). The HTML loads the IBM Plex
# fonts from Google Fonts, so the export needs network access (or the fonts installed locally).

set -euo pipefail

ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT (see configs/paths.env.example)}
IN="${1:-$ROOT/analysis/figure_sources/method_overview.html}"
OUT="${2:-${FIG_DIR:-$ROOT/analysis/figures}/method.pdf}"
MARGIN="${3:-1}"

CHROME="${CHROME:-$(command -v google-chrome || command -v chromium || command -v chromium-browser || true)}"
[[ -n "$CHROME" && -x "$CHROME" ]] || { echo "Chrome/Chromium not found; set CHROME to its binary" >&2; exit 1; }
command -v gs >/dev/null || { echo "ghostscript (gs) not found" >&2; exit 1; }

case "$IN" in /*) ;; *) IN="$PWD/$IN" ;; esac
TMPBASE="$(mktemp "${TMPDIR:-/tmp}/figure_raw.XXXXXX")"
TMP="$TMPBASE.pdf"
PROFILE="$(mktemp -d "${TMPDIR:-/tmp}/chrome_profile.XXXXXX")"
trap 'rm -rf "$TMPBASE" "$TMP" "$PROFILE"' EXIT

# Header/footer suppression comes from the page's @page { margin: 0 } rule, not a flag.
# Recent Chrome builds write the PDF and then hang on shutdown, so run it in the
# background, wait until the PDF exists with a stable size, and terminate it ourselves.
# A fresh --user-data-dir avoids contending with a desktop Chrome profile.
"$CHROME" --headless --disable-gpu --user-data-dir="$PROFILE" \
  --print-to-pdf="$TMP" "file://$IN" >/dev/null 2>&1 &
CPID=$!
DEADLINE=$((SECONDS + 60))
while (( SECONDS < DEADLINE )); do
  if [[ -s "$TMP" ]]; then
    S1="$(wc -c < "$TMP")"; sleep 0.5; S2="$(wc -c < "$TMP")"
    [[ "$S1" == "$S2" ]] && break
  fi
  sleep 0.5
done
kill "$CPID" 2>/dev/null || true
pkill -9 -f -- "--user-data-dir=$PROFILE" 2>/dev/null || true
wait "$CPID" 2>/dev/null || true

[[ -s "$TMP" ]] || { echo "Chrome produced no PDF within 60s for $IN" >&2; exit 1; }

PAGES="$(gs -q -dNODISPLAY -dNOSAFER -c "($TMP) (r) file runpdfbegin pdfpagecount = quit")"
if [[ "$PAGES" != "1" ]]; then
  echo "warning: Chrome produced $PAGES pages; enlarge the @page size in the HTML so the figure fits on one" >&2
fi

read -r _ X0 Y0 X1 Y1 < <(gs -o - -sDEVICE=bbox "$TMP" 2>&1 | grep '%%HiResBoundingBox' | head -1)

W="$(python3 -c "print(round($X1-$X0+2*$MARGIN, 2))")"
H="$(python3 -c "print(round($Y1-$Y0+2*$MARGIN, 2))")"
OX="$(python3 -c "print(round(-($X0-$MARGIN), 2))")"
OY="$(python3 -c "print(round(-($Y0-$MARGIN), 2))")"

mkdir -p "$(dirname "$OUT")"
gs -dQUIET -dBATCH -dNOPAUSE -sDEVICE=pdfwrite \
   -dDEVICEWIDTHPOINTS="$W" -dDEVICEHEIGHTPOINTS="$H" -dFIXEDMEDIA \
   -o "$OUT" -c "<</PageOffset [$OX $OY]>> setpagedevice" -f "$TMP"

echo "wrote $OUT (${W} x ${H} pt, ${MARGIN}pt margin)"
