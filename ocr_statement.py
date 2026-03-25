"""Subprocess worker: DocTR-based statement PDF OCR.

Usage:
    python ocr_statement.py <pdf_path> <poppler_path>

Prints JSON to stdout: list of page dicts with keys:
    page_number, raw_ocr_words (list of {text, confidence, x_min, y_min, x_max, y_max, has_plus_prefix}), status

Credit detection uses Geometric Welding: for each amount token, a narrow
pixel strip immediately to its left is probed for non-background marks
(the '+' sign that DocTR fails to recognise as text).  This approach is
hardware-agnostic — works on color, greyscale, and messy scans.
"""
import os
import sys
import gc
import json
import re
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Matches amount-like tokens: 1,234.56  +450.00  -1,234.56  etc.
# Leading +/- is optional so DocTR-merged tokens like "+7,627.00" are recognised.
_AMT_RE = re.compile(r"^[+\-]?\d[\d,]*\.\d{2}$")

# Cleans up amounts where DocTR merges trailing characters:
#   "7,627.002" or "7,627.001" → "7,627.00"  (trailing row-index digit)
#   "7,627.0023" → "7,627.00"  (trailing multi-digit row-index e.g. "23")
#   "3,598.26R" → "3,598.26"  (trailing credit marker 'R')
_TRAIL_CLEANUP = re.compile(r'^([+\-]?\d[\d,]*\.\d{2})(\d{1,3}|[Rr])$')

# Cleans up dates where DocTR merges the trailing row-index digit or table border:
#   "25/02/20261" → "25/02/2026", "25/02/2026]" → "25/02/2026"
_DATE_TRAIL_CLEANUP = re.compile(r'^(\d{2}[/\-]\d{2}[/\-]\d{2,4})[\d\]Il|]{1,3}$')

# ---------------------------------------------------------------------------
# DocTR engine (one-time init per process)
# ---------------------------------------------------------------------------
_doctr_engine = None

def _get_doctr():
    global _doctr_engine
    if _doctr_engine is None:
        print("Loading DocTR model (first run may download weights)...", file=sys.stderr, flush=True)
        from doctr.models import ocr_predictor
        _doctr_engine = ocr_predictor(pretrained=True)
        print("DocTR model loaded.", file=sys.stderr, flush=True)
    return _doctr_engine


def _has_plus_prefix(img_rgb: np.ndarray, x_min: float, y_min: float,
                     y_max: float) -> bool:
    """Geometric Welding: probe a narrow strip left of an amount token for
    a '+' sign that DocTR missed.

    Layout (normalised X, measured from real statements):
        +  sign sits at x_min − 0.030 … x_min
        ₹  symbol may also sit in this zone

    We probe the full 0–3% strip left of x_min and check specifically
    for the CROSS pattern of '+' (horizontal + vertical strokes) rather
    than just "any dark mark".  This distinguishes '+' from '₹' which
    has horizontal bars but a different vertical structure.
    """
    h, w = img_rgb.shape[:2]

    # Probe the full strip 0–3% left of x_min
    probe_x2 = int(x_min * w)
    probe_x1 = max(0, int((x_min - 0.030) * w))
    probe_y1 = int(y_min * h) + 2
    probe_y2 = int(y_max * h) - 2

    pw = probe_x2 - probe_x1
    ph = probe_y2 - probe_y1
    if pw < 4 or ph < 4:
        return False

    patch = img_rgb[probe_y1:probe_y2, probe_x1:probe_x2]
    if patch.size == 0:
        return False

    # Binary mark mask (any pixel darker than 220 = ink)
    is_mark = np.any(patch < 220, axis=2)
    total_mark = np.sum(is_mark)
    if total_mark < 3:
        return False  # no marks at all

    # --- Cross-pattern detection for '+' ---
    # A '+' sign has:
    #   1. A horizontal stroke in the vertical center (~40-60% of height)
    #   2. A vertical stroke in the horizontal center (~30-70% of width)
    #
    # The ₹ symbol has horizontal bars but its vertical component is
    # off-center (a curved stroke on the left).  By requiring marks
    # in BOTH the horizontal-center and vertical-center, we detect '+'.

    # Horizontal stroke: project marks onto rows → find rows with marks
    row_marks = np.sum(is_mark, axis=1)  # marks per row
    # Vertical stroke: project marks onto columns → find cols with marks
    col_marks = np.sum(is_mark, axis=0)  # marks per column

    # '+' horizontal stroke → a row near the vertical center has marks
    # spanning at least 30% of the patch width
    mid_y_lo = int(ph * 0.25)
    mid_y_hi = int(ph * 0.75)
    has_h_stroke = any(
        row_marks[r] >= pw * 0.20
        for r in range(mid_y_lo, min(mid_y_hi + 1, ph))
    )

    # '+' vertical stroke → a column near the horizontal center has marks
    # spanning at least 30% of the patch height
    mid_x_lo = int(pw * 0.15)
    mid_x_hi = int(pw * 0.85)
    has_v_stroke = any(
        col_marks[c] >= ph * 0.20
        for c in range(mid_x_lo, min(mid_x_hi + 1, pw))
    )

    return has_h_stroke and has_v_stroke


# ---------------------------------------------------------------------------
# Statement PDF → OCR words
# ---------------------------------------------------------------------------

def process_statement_pdf(pdf_path: str, poppler_path: str = None) -> list[dict]:
    from pdf2image import convert_from_path

    pages_out = []
    try:
        kwargs = {"dpi": 350}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        pil_images = convert_from_path(pdf_path, **kwargs)
    except Exception as e:
        return [{"page_number": 1, "raw_ocr_words": [], "status": f"failed: {e}"}]

    model = _get_doctr()

    # Process ONE page at a time — lower peak memory, prevents OOM
    for page_idx, pil_img in enumerate(pil_images):
        page_img = np.array(pil_img)
        del pil_img

        try:
            result = model([page_img])
        except Exception as e:
            pages_out.append({"page_number": page_idx + 1, "raw_ocr_words": [],
                              "status": f"failed: {e}"})
            continue

        page = result.pages[0]
        words = []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    conf = float(word.confidence)
                    if conf < 0.15:   # was 0.4 — digits on scanned PDFs often score 0.2–0.35
                        continue
                    (x_min, y_min), (x_max, y_max) = word.geometry

                    raw_text = word.value.strip()

                    # ── Pre-clean dates ──────────────────────────────────────
                    # DocTR sometimes merges the row-index digit into the date
                    # if they are printed too close together (e.g. "25/02/20261")
                    _dtc = _DATE_TRAIL_CLEANUP.match(raw_text)
                    if _dtc:
                        raw_text = _dtc.group(1)

                    # ── Pre-clean amount tokens ──────────────────────────────
                    # DocTR sometimes merges adjacent characters into one token:
                    #  (a) Trailing digit: row-index digit merged into amount
                    #      e.g. "7,627.002" → "7,627.00"
                    #  (b) Trailing "R": credit marker merged into amount column is read as part of the
                    #      amount, producing "7,627.002" instead of "7,627.00".
                    #      Detect: token looks like a valid amount PLUS one extra
                    #      trailing digit that is NOT part of the 2-decimal format.
                    #
                    #  (b) Trailing "R" credit marker: the statement prints "R"
                    #      immediately after credit amounts ("3,598.26R").
                    #      Strip "R" and set the credit flag.
                    #
                    # We handle this BEFORE _AMT_RE so downstream code always
                    # sees a clean "D,DDD.DD" amount token.
                    _tc = _TRAIL_CLEANUP.match(raw_text)
                    if _tc:
                        raw_text = _tc.group(1)          # strip trailing char
                        if _tc.group(2).upper() == 'R':
                            # "R" = credit indicator on this statement; we will
                            # propagate this as has_plus_prefix below.
                            pass  # handled after plus_prefix logic below

                    # --- Strip OCR artifacts for ₹ symbol prepended to amounts ---
                    # DocTR frequently reads the '₹' symbol as 'R', '?', or literally '₹'.
                    # This happens right before the numeric amount. If we leave it, the LLM
                    # or downstream regex might misparse it.
                    # Note: We cannot safely strip '2' because it's ambiguous with real amounts.
                    _PREFIX_CLEANUP = re.compile(r'^[R\?₹]+')
                    if re.match(r'^[R\?₹]+[+\-]?\d[\d,]*\.\d{2}$', raw_text):
                        raw_text = _PREFIX_CLEANUP.sub('', raw_text)

                    # --- Credit detection ---
                    # Strategy 1: DocTR includes '+' directly in the token value
                    # (e.g. "+7,627.00").  Strip it and mark as credit.
                    plus_prefix = False
                    # Also mark credit if the original token had trailing "R"
                    if _tc and _tc.group(2).upper() == 'R':
                        plus_prefix = True
                    if raw_text.startswith("+") and _AMT_RE.match(raw_text):
                        plus_prefix = True
                        raw_text = raw_text[1:]  # strip leading '+' for clean numeric text

                    # Strategy 2: Geometric Welding — DISABLED.
                    # Previously probed dark pixels to the left of amount tokens,
                    # but this causes false positives on different PDF layouts
                    # where borders, text, or other marks exist near amounts.
                    # Credit detection now relies solely on:
                    #   - Explicit '+' prefix in token text
                    #   - Trailing 'R' credit marker on the statement
                    # if not plus_prefix and page_img is not None and _AMT_RE.match(raw_text):
                    #     plus_prefix = _has_plus_prefix(page_img, x_min, y_min, y_max)

                    words.append({
                        "text": raw_text,
                        "confidence": conf,
                        "x_min": float(x_min),
                        "y_min": float(y_min),
                        "x_max": float(x_max),
                        "y_max": float(y_max),
                        "low_confidence": bool(conf < 0.6),
                        "has_plus_prefix": bool(plus_prefix),
                    })
        pages_out.append({
            "page_number": page_idx + 1,
            "raw_ocr_words": words,
            "status": "success" if words else "failed: no words detected",
        })

        del page_img, result
        gc.collect()

    return pages_out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps([{"error": "Usage: python ocr_statement.py <pdf_path> [poppler_path]"}]))
        sys.exit(1)

    pdf_path = sys.argv[1]
    poppler_path = sys.argv[2] if len(sys.argv) >= 3 else None
    pages = process_statement_pdf(pdf_path, poppler_path)
    print(json.dumps(pages, ensure_ascii=False))
