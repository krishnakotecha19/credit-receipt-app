"""Subprocess worker: DocTR-based statement PDF OCR.

Usage:
    python ocr_statement.py <pdf_path> <poppler_path>

Prints JSON to stdout: list of page dicts with keys:
    page_number, raw_ocr_words (list of {text, confidence, x_min, y_min, x_max, y_max, has_plus_prefix}), status

Credit detection uses Geometric Welding: for each amount token, a narrow
pixel strip immediately to its left is probed for non-background marks
(the '+' sign that DocTR fails to recognise as text).  This approach is
hardware-agnostic — works on color, greyscale, and messy scanssssssssssss.
"""
import os
import sys
import gc
import json
import re
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

_OCR_STMT_VERSION = "v7-green-color-credit"
print(f"[ocr_statement] Loaded {_OCR_STMT_VERSION}", file=sys.stderr, flush=True)

# Matches amount-like tokens: 1,234.56  +450.00  -1,234.56  etc.
# Leading +/- is optional so DocTR-merged tokens like "+7,627.00" are recognised.
_AMT_RE = re.compile(r"^[+\-]?\d[\d,]*\.\d{2}$")

# Cleans up amounts where DocTR merges trailing row-index digit(s):
#   "7,627.002" → "7,627.00"   (trailing '2')
#   "7,627.0023" → "7,627.00"  (trailing '23')
_TRAIL_CLEANUP = re.compile(r'^([+\-]?\d[\d,]*\.\d{2})(\d{1,3})$')

# Detects trailing 'R' credit marker on amounts:
#   "3,598.26R" → amount "3,598.26", is_credit=True
# The 'R' (Reversal/Refund) suffix is used on credit card statements for credits.
_TRAIL_R_CREDIT = re.compile(r'^([+\-]?\d[\d,]*\.\d{2})[Rr]$')

# Cleans up dates where DocTR merges the trailing row-index digit or table border:
#   "25/02/20261" → "25/02/2026", "25/02/2026]" → "25/02/2026"
_DATE_TRAIL_CLEANUP = re.compile(r'^(\d{2}[/\-]\d{2}[/\-]\d{2,4})[\d\]Il|]{1,3}$')

# ---------------------------------------------------------------------------
# ₹ prefix artifact patterns
# ---------------------------------------------------------------------------
# DocTR misreads the ₹ glyph in several ways:
#   (a) As a letter: R, B, F, E, ?, or the literal ₹ character
#   (b) As a digit: most commonly 2 or 7, but also 8, 9, 6, etc.
#
# For (a) the prefix is unambiguously non-numeric — safe to strip outright.
# For (b) the prefix digit is ambiguous against real digits.  We use the
#   Indian number format rule to detect it:
#
#   ** Indian bank statements ALWAYS format amounts ≥ 1000 with a comma:
#      1,234.56   7,367.00   not   1234.56   7367.00
#
#   So a token whose integer part has 4+ digits with NO comma is structurally
#   invalid — the first digit is the ₹ glyph.
#   Stripping it leaves a valid 1-3 digit amount (or 4+ commaed amount).
#
# Pattern: strip any leading char that, once removed, leaves a valid amount
# AND (original had no comma in 4+ integer digits).
_NOCOMMA_4DIGIT_RE = re.compile(r'^[+\-]?(\d)(\d{3}\.\d{2})$')  # e.g. 7367.00 → 367.00
_NOCOMMA_5DIGIT_RE = re.compile(r'^[+\-]?(\d)(\d{4}\.\d{2})$')  # e.g. 71234.00 → 1234.00 (then needs comma check)
_LETTER_PREFIX_RE  = re.compile(r'^[RBFEb?₹]+(\d[\d,]*\.\d{2})$')

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


def _is_green_text(img_rgb: np.ndarray, x_min: float, y_min: float,
                   x_max: float, y_max: float) -> bool:
    """Detect if a word's text is printed in GREEN color.

    Credit card statements (e.g. HDFC) print credit amounts in green
    and debit amounts in black.  DocTR doesn't detect color, so we
    sample the pixel colors inside the word bounding box and check
    if the ink is green-dominant.

    Green text criteria:
      - G channel > R channel by at least 30
      - G channel > B channel by at least 30
      - Pixel is actually ink (not white background): max(R,G,B) < 220
    """
    h, w = img_rgb.shape[:2]

    y1 = max(0, int(y_min * h))
    y2 = min(h, int(y_max * h))
    x1 = max(0, int(x_min * w))
    x2 = min(w, int(x_max * w))

    if y2 - y1 < 2 or x2 - x1 < 2:
        return False

    patch = img_rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return False

    r = patch[:, :, 0].astype(np.int16)
    g = patch[:, :, 1].astype(np.int16)
    b = patch[:, :, 2].astype(np.int16)

    # Ink pixels: not white (at least one channel < 220)
    is_ink = np.max(patch, axis=2) < 220

    if np.sum(is_ink) < 3:
        return False  # not enough ink pixels

    # Green-dominant: G > R + 30 AND G > B + 30 among ink pixels
    is_green = is_ink & (g > r + 30) & (g > b + 30)

    green_ratio = np.sum(is_green) / max(np.sum(is_ink), 1)
    return green_ratio > 0.3  # at least 30% of ink pixels are green


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
                        raw_text = _tc.group(1)   # strip trailing row-index digit(s)

                    # --- Strip OCR artifacts for ₹ symbol prepended to amounts ---
                    # DocTR reads the ₹ glyph in two ways, both produce bogus leading chars:
                    #
                    # (a) Letter variant: R, B, F, E, ?, ₹          → strip outright
                    # (b) Digit variant:  2, 7, 8, 9, 6, etc.        → strip via structural rule
                    #
                    # Structural rule for digit variants:
                    #   Indian bank statements ALWAYS comma-format amounts ≥ 1000:
                    #     real 7367 rupees → printed as "7,367.00"
                    #   So a 4-digit no-comma integer in an amount token is ALWAYS bogus.
                    #   Stripping the leading digit recovers the true amount.
                    #   This handles 7367.00→367.00, 2367.00→367.00, etc.

                    rupee_prefix_stripped = False

                    # (a) Explicit non-numeric ₹ prefix (R, B, F, E, ?, ₹)
                    _lm = _LETTER_PREFIX_RE.match(raw_text)
                    if _lm:
                        raw_text = _lm.group(1)
                        rupee_prefix_stripped = True

                    # (b) Digit prefix via no-comma 4-digit rule
                    #   e.g. "7367.00" → integer part "7367" has 4 digits, no comma → strip "7" → "367.00"
                    if not rupee_prefix_stripped:
                        _sgn = ""
                        _chk = raw_text
                        if _chk[:1] in ('+', '-'):
                            _sgn = _chk[0]
                            _chk = _chk[1:]
                        _nm = _NOCOMMA_4DIGIT_RE.match(_sgn + _chk)
                        if _nm:
                            # 4-digit no-comma integer — first digit is ₹ artifact
                            raw_text = _sgn + _nm.group(2)
                            rupee_prefix_stripped = True
                        elif not rupee_prefix_stripped:
                            # 5-digit no-comma: XNNNN.DD → if removing first digit gives
                            # a comma-formatted amount that makes sense (e.g. 71234.00 → 1,234 requires comma)
                            # Only strip if remainder starts by forming a comma-insertable valid amount
                            _nm5 = _NOCOMMA_5DIGIT_RE.match(_sgn + _chk)
                            if _nm5:
                                remainder = _nm5.group(2)  # 4 digits + .DD, e.g. "1234.00"
                                # Real 4-digit amounts are always printed with comma: 1,234
                                # so a bare 4-digit chunk is also bogus
                                raw_text = _sgn + remainder
                                rupee_prefix_stripped = True

                    # --- Credit detection ---
                    # Two reliable signals:
                    #   1. Explicit '+' prefix in the token text (+7,627.00)
                    #   2. Trailing 'R' suffix (3,598.26R) — credit card refund marker
                    # Geometric welding (pixel probing) is DISABLED — it caused
                    # false positives on different PDF layouts.
                    plus_prefix = False

                    # Check trailing 'R' credit marker (e.g. "3,598.26R")
                    _rc = _TRAIL_R_CREDIT.match(raw_text)
                    if _rc:
                        raw_text = _rc.group(1)   # strip trailing R
                        plus_prefix = True

                    # Check explicit '+' prefix (e.g. "+7,627.00")
                    if raw_text.startswith("+") and _AMT_RE.match(raw_text):
                        plus_prefix = True
                        raw_text = raw_text[1:]  # strip leading '+' for clean numeric text

                    # Strategy 2: Green text color detection.
                    # Credit card statements (e.g. HDFC) print credit amounts
                    # in GREEN and debit amounts in BLACK. DocTR can't read
                    # color, so we sample the pixel colors of the amount token.
                    if not plus_prefix and page_img is not None and _AMT_RE.match(raw_text):
                        plus_prefix = _is_green_text(
                            page_img, x_min, y_min, x_max, y_max
                        )

                    words.append({
                        "text": raw_text,
                        "confidence": conf,
                        "x_min": float(x_min),
                        "y_min": float(y_min),
                        "x_max": float(x_max),
                        "y_max": float(y_max),
                        "low_confidence": bool(conf < 0.6),
                        "has_plus_prefix": bool(plus_prefix),
                        "rupee_prefix_stripped": bool(rupee_prefix_stripped),
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
