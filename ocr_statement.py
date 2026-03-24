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

# Matches amount-like tokens: 1,234.56  450.00  etc.
_AMT_RE = re.compile(r"^\d[\d,]*\.\d{2}$")

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
        +  sign sits at x_min − 0.020 … x_min − 0.008
        ₹  symbol sits at x_min − 0.005 … x_min − 0.002
        amount digits start at x_min

    We probe the '+' zone (0.5–2.0% left of x_min), skipping the
    immediate ₹ symbol zone to avoid false positives.

    This is agnostic to color — it detects ANY mark (black, green, grey)
    against a white/light background.
    """
    h, w = img_rgb.shape[:2]

    # Probe strip: skip the ₹ zone (0–0.5%) and check the + zone (0.5–2.0%)
    probe_x2 = int((x_min - 0.005) * w)
    probe_x1 = max(0, int((x_min - 0.020) * w))
    # Vertically: inside the word bbox with 2px inset to avoid row borders
    probe_y1 = int(y_min * h) + 2
    probe_y2 = int(y_max * h) - 2

    if probe_x1 >= probe_x2 or probe_y1 >= probe_y2:
        return False

    patch = img_rgb[probe_y1:probe_y2, probe_x1:probe_x2]
    if patch.size == 0:
        return False

    # A mark pixel = any pixel where at least one RGB channel is below 220
    # (white background is ~255,255,255; any ink/print mark will be darker)
    is_mark = np.any(patch < 220, axis=2)
    mark_ratio = np.sum(is_mark) / is_mark.size

    return mark_ratio > 0.02


# ---------------------------------------------------------------------------
# Statement PDF → OCR words
# ---------------------------------------------------------------------------

def process_statement_pdf(pdf_path: str, poppler_path: str = None) -> list[dict]:
    from pdf2image import convert_from_path

    pages_out = []
    try:
        kwargs = {"dpi": 200}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        pil_images = convert_from_path(pdf_path, **kwargs)
    except Exception as e:
        return [{"page_number": 1, "raw_ocr_words": [], "status": f"failed: {e}"}]

    # Keep RGB copies for geometric welding (plus-sign detection)
    rgb_images = [np.array(img) for img in pil_images]
    del pil_images
    gc.collect()

    try:
        model = _get_doctr()
        result = model(rgb_images)
    except Exception as e:
        return [{"page_number": i + 1, "raw_ocr_words": [], "status": f"failed: {e}"}
                for i in range(len(rgb_images))]

    for page_idx, page in enumerate(result.pages):
        page_img = rgb_images[page_idx] if page_idx < len(rgb_images) else None
        words = []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    conf = float(word.confidence)
                    if conf < 0.4:
                        continue
                    (x_min, y_min), (x_max, y_max) = word.geometry

                    # Geometric Welding: for amount-like tokens, probe the
                    # pixel strip to the left for a '+' sign mark.
                    plus_prefix = False
                    if page_img is not None and _AMT_RE.match(word.value):
                        plus_prefix = _has_plus_prefix(
                            page_img, x_min, y_min, y_max
                        )

                    words.append({
                        "text": word.value,
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

    del result, rgb_images
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
