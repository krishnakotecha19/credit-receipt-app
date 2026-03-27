"""Subprocess worker: PaddleOCR-based receipt extraction.

Usage:
    python ocr_receipt.py <image_path><xydvfdz>

Prints JSON to stdout with keys:
    receipt_file, vendor, amount, date, raw_text, confidence, status
"""
import os
import sys
import re
import gc
import json
import base64
import cv2
import numpy as np
import requests
from dateutil import parser as dateutil_parser
from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

MIN_IMAGE_WIDTH = 800

# ---------------------------------------------------------------------------
# PaddleOCR engine (one-time init per process)
# ---------------------------------------------------------------------------
_ocr_engine = None

def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        import paddle
        _use_gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        if _use_gpu:
            print("PaddleOCR: Using GPU (CUDA)", file=sys.stderr, flush=True)
        else:
            print("PaddleOCR: Using CPU", file=sys.stderr, flush=True)
        _ocr_engine = PaddleOCR(
            lang="en",
            ocr_version="PP-OCRv4",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
            device="gpu" if _use_gpu else "cpu",
        )
    return _ocr_engine

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _fix_exif_orientation(img: np.ndarray, image_source) -> np.ndarray:
    """Rotate image according to EXIF orientation tag so vertical photos stay vertical.
    image_source can be a file path (str) or raw bytes."""
    try:
        from PIL import Image
        import io
        if isinstance(image_source, (bytes, bytearray)):
            pil = Image.open(io.BytesIO(image_source))
        else:
            pil = Image.open(image_source)
        exif = pil.getexif()
        orientation = exif.get(274)  # 274 = Orientation tag
        if orientation == 3:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif orientation == 6:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass
    return img


def _resize_image(img: np.ndarray) -> np.ndarray:
    """Resize to meet MIN_IMAGE_WIDTH / MAX_SIDE constraints."""
    MAX_SIDE = 4000
    h, w = img.shape[:2]
    if w < MIN_IMAGE_WIDTH:
        scale = MIN_IMAGE_WIDTH / w
        new_h = int(h * scale)
        if new_h > MAX_SIDE:
            scale = MAX_SIDE / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif h > MAX_SIDE or w > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def preprocess_image(image_path: str) -> np.ndarray:
    """Light preprocessing: fix EXIF rotation, resize if needed, keep original colors.
    PaddleOCR v4 handles raw images well — heavy thresholding destroys
    text on colored/gradient backgrounds.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = _fix_exif_orientation(img, image_path)
    return _resize_image(img)


def preprocess_image_bytes(img_bytes: bytes) -> np.ndarray:
    """Preprocess an image from raw bytes (no disk I/O).
    Uses cv2.imdecode instead of cv2.imread."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image from bytes")
    img = _fix_exif_orientation(img, img_bytes)
    return _resize_image(img)



# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

# AMOUNT regex: handles all forms of monetary amounts.
# Group 1: currency-prefixed (₹1200, Rs. 1,200.00, INR 450)
# Group 2: with commas, no prefix (1,200 or 18,999 or 1,200.00)
# Group 3: 3+ digits without commas (1200, 18999, 450.00) — needs \b to avoid partial matches
_AMOUNT_RE = re.compile(
    r"(?:[\₹]|Rs\.?|INR)\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)"
    r"|"
    r"\b(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?)\b"
    r"|"
    r"\b(\d{3,8}(?:\.\d{1,2})?)\b",          # 3-8 digits max (up to ₹9,99,99,999)
    re.IGNORECASE,
)

# Sane limits: no receipt amount should be below ₹1 or above ₹1 crore
_MIN_AMOUNT = 1.0
_MAX_AMOUNT = 10_000_000.0  # 1 crore

def _clean_amount(raw: str) -> float | None:
    """Remove ₹, Rs, commas and convert to float.  Rejects insane values."""
    cleaned = raw.replace("₹", "").replace("Rs", "").replace("Rs.", "")
    cleaned = cleaned.replace(",", "").strip()
    try:
        val = float(cleaned)
        if val < _MIN_AMOUNT or val > _MAX_AMOUNT:
            return None
        return val
    except ValueError:
        return None

# Lines to EXCLUDE before keyword matching (qty/quantity totals, not amounts)
_NOT_AMOUNT_LINE = re.compile(
    r"tota[l1]?\s*qty|tota[l1]?\s*quantity|qty\s*tota[l1]?|item\s*count|"
    r"tota[l1]?\s*items?|no\.?\s*of\s*items?",
    re.IGNORECASE,
)

# Lines that are SUB-TOTALS — not the final payable amount
_SUBTOTAL_LINE = re.compile(
    r"sub\s*[-_]?\s*tota[l1]?|subtota[l1]?|item\s*tota[l1]?|product\s*tota[l1]?|"
    r"order\s*sub|before\s*tax|excl\.?\s*(tax|gst)",
    re.IGNORECASE,
)

# Tax/discount lines — never the final amount
# Also catches OCR-merged forms like "TotaGST5%CGST2.5%+SGST"
_TAX_DISCOUNT_LINE = re.compile(
    r"c?s?gst|igst|vat|\btax\b|\bcess\b|surcharge|d[il]scount|coupon|promo|"
    r"savings?|you\s*saved?|cashback|round\s*off|rounding",
    re.IGNORECASE,
)

# Tier 1: Very strong indicators of the exact final payable amount
# NOTE: OCR frequently truncates or merges words (e.g. "GrandTota" instead
# of "Grand Total", "TotalAmt" merged).  Use `tota[l1]?` to tolerate a
# missing or mis-read trailing 'l', and allow optional whitespace between
# words so merged forms like "GrandTotal" / "GrandTota" both match.
_STRICT_TOTAL_KEYWORDS = re.compile(
    r"grand\s*tota[l1]?|net\s*tota[l1]?|net\s*amount|net\s*payable|"
    r"net\s*rs\.?|tota[l1]?\s*rs\.?|"                                  # "Net Rs", "Total RS"
    r"tota[l1]?\s*payable|payable\s*amount|tota[l1]?\s*amount|amount\s*payable|"
    r"amount\s*paid|tota[l1]?\s*paid|tota[l1]?\s*due|amount\s*due|"
    r"you\s*pay|to\s*pay|net\s*pay|bill\s*tota[l1]?|"
    r"tota[l1]?\s*payment|payment\s*amount|final\s*amount|final\s*tota[l1]?|"
    r"balance\s*due|invoice\s*amount|charged|debit",
    re.IGNORECASE,
)

# Tier 2: General keywords — may catch subtotal/tax if not filtered
_TOTAL_KEYWORDS = re.compile(
    r"\btota[l1]?\b|net\s*amt|payable|paid|"
    r"bill\s*amount|invoice\s*tota[l1]?|balance|tender\s*amount|"
    r"payment\s*successful|gross\s*amount|payment",
    re.IGNORECASE,
)

# Fallback keywords — only used when no primary keyword finds an amount
_FALLBACK_KEYWORDS = re.compile(
    r"\bamt\b|amt\s*[:.]",
    re.IGNORECASE,
)

# Words that disqualify a line from being a vendor name
_NON_VENDOR_WORDS = {"total", "tax", "invoice", "amount", "gst", "cgst", "sgst",
                      "subtotal", "sub-total", "grand", "payment", "change",
                      "cash", "card", "date", "bill", "receipt", "qty", "quantity",
                      "successful", "failed", "pending"}

# Lines containing these words should NOT have their numbers treated as amounts
_NON_AMOUNT_KEYWORDS = re.compile(
    r"txn\s*id|order\s*id|ref|rrn|serial|mid\b|tid\b|card\s*no|"
    r"auth.?code|phone|mobile|gstin|hsn|pin|otp|"
    r"\baid\b|\brrn\b",
    re.IGNORECASE,
)

# Txn ID / Order ID patterns that embed a date as YYYYMMDD
_TXN_DATE_RE = re.compile(r"(20[2-3]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")


def _extract_amounts_from_line(line: str) -> list[float]:
    """Extract all amounts from a single line using the standard regex."""
    amounts = []
    for m in _AMOUNT_RE.finditer(line):
        raw = m.group(1) or m.group(2) or m.group(3)
        if raw:
            val = _clean_amount(raw)
            if val is not None:
                amounts.append(val)
    return amounts


# Lenient amount regex — no \b word-boundary requirement.
# Used ONLY when a keyword is found on a line but _extract_amounts_from_line
# returns nothing (OCR merged keyword + amount, e.g. "GrandTota340.00").
_LENIENT_AMOUNT_RE = re.compile(
    r"(?:[\₹]|Rs\.?|INR)\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)"
    r"|(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?)"
    r"|(\d{3,8}(?:\.\d{1,2})?)",
    re.IGNORECASE,
)


def _extract_amounts_lenient(line: str) -> list[float]:
    """Like _extract_amounts_from_line but without word-boundary guards.

    Only call this when a keyword was already confirmed on the line — the
    relaxed matching would cause too many false positives on arbitrary text.
    """
    amounts = []
    for m in _LENIENT_AMOUNT_RE.finditer(line):
        raw = m.group(1) or m.group(2) or m.group(3)
        if raw:
            val = _clean_amount(raw)
            if val is not None:
                amounts.append(val)
    return amounts


# Regex to detect currency-prefixed amounts on a line
_CURRENCY_PREFIX_RE = re.compile(
    r"(?:[\₹]|Rs\.?|INR)\s*\d", re.IGNORECASE
)


def _parse_amount(lines: list[str]):
    """Extract the total/final payable amount from receipt lines.

    Two-strategy approach based on receipt type:

    1. **Default (bottom-up):** Most receipts list items first, then subtotals,
       then the final total near the bottom. We scan bottom-up and return the
       FIRST keyword-matched amount we find (i.e., the one closest to the
       bottom of the receipt — the true final amount).

    2. **"Payment Successful" (Paytm edge-case, top-down):** Paytm receipts
       show the paid amount ABOVE a "Payment Successful" line. We scan
       top-down and return the FIRST keyword-matched amount found.

    Keyword priority (multi-pass, same direction each pass):
        Pass 1: Strict keywords (Grand Total, Net Amount, Total Payable …)
        Pass 2: Primary keywords (total, paid, payment …)
        Pass 3: Fallback keywords (amt)
        Pass 4: Currency-prefixed amounts (₹ / Rs / INR)
        Pass 5: Standalone numbers (entire line is just a number)

    All existing filters (subtotal, tax/discount, ID-line exclusions) remain.
    Returns None if nothing is found.
    """

    # ── Step 1: detect Paytm "Payment Successful" edge-case ──────────
    _PAYMENT_SUCCESS_RE = re.compile(r"payment\s+success", re.IGNORECASE)
    is_payment_success_receipt = any(
        _PAYMENT_SUCCESS_RE.search(line) for line in lines
    )

    # ── Step 2: build ordered (index, line) pairs ────────────────────
    indexed_lines = list(enumerate(lines))
    if not is_payment_success_receipt:
        # Bottom-up: reverse so we scan from the end of the receipt first
        indexed_lines = indexed_lines[::-1]
    # For Payment Successful: keep top-down order (default enumerate order)

    # ── Helper: single-pass scan in the chosen direction ─────────────
    def _scan_for_amount(keyword_tier: str):
        """Scan indexed_lines once and return the first amount matching the
        requested keyword tier, using ORIGINAL receipt order for adjacent-
        line lookups.

        keyword_tier: 'strict' | 'primary' | 'fallback' | 'currency'

        When OCR splits a keyword and its value into separate text boxes
        (e.g. "Grand Total" and "550" as two entries), the amount always
        follows the keyword in original reading order — NOT in scan order.
        Previous logic used prev_had_keyword which looked forward in scan
        direction, breaking bottom-to-top scans.
        """

        for scan_pos, (orig_idx, line) in enumerate(indexed_lines):
            is_id_line = bool(_NON_AMOUNT_KEYWORDS.search(line))
            is_subtotal = bool(_SUBTOTAL_LINE.search(line))
            is_tax_discount = bool(_TAX_DISCOUNT_LINE.search(line))
            is_not_amount = bool(_NOT_AMOUNT_LINE.search(line))

            # For Payment Successful receipts, stop once we hit the success line
            if is_payment_success_receipt and _PAYMENT_SUCCESS_RE.search(line):
                pass  # fall through to normal logic, but mark to stop after

            line_amounts = _extract_amounts_from_line(line)

            # Determine if this line matches the requested keyword tier
            has_keyword = False
            if keyword_tier == 'strict':
                has_keyword = (bool(_STRICT_TOTAL_KEYWORDS.search(line))
                               and not is_not_amount and not is_subtotal)
            elif keyword_tier == 'primary':
                has_keyword = (bool(_TOTAL_KEYWORDS.search(line))
                               and not is_not_amount and not is_subtotal
                               and not is_tax_discount)
            elif keyword_tier == 'fallback':
                has_keyword = bool(_FALLBACK_KEYWORDS.search(line))
            elif keyword_tier == 'currency':
                has_keyword = (bool(_CURRENCY_PREFIX_RE.search(line))
                               and not is_subtotal and not is_tax_discount)

            if has_keyword:
                # Case A: amount is on the SAME line as the keyword
                if line_amounts and not is_id_line:
                    if keyword_tier == 'primary' and (is_subtotal or is_tax_discount):
                        continue
                    return line_amounts[-1]

                # Case A2: keyword + amount MERGED into one OCR box without
                # whitespace (e.g. "GrandTota340.00") — standard regex fails
                # because \b doesn't fire between a letter and a digit when
                # both are word characters.  Use lenient extraction.
                if not is_id_line:
                    lenient = _extract_amounts_lenient(line)
                    if lenient:
                        if keyword_tier == 'primary' and (is_subtotal or is_tax_discount):
                            continue
                        return lenient[-1]

                # Case B: keyword and amount are on SEPARATE OCR boxes.
                # Look at ±2 lines in ORIGINAL receipt order.  Kept small to
                # avoid grabbing OCR garbage; wider columnar gaps are handled
                # by the "largest standalone amount" fallback in Step 4.5.
                for adj_offset in [1, -1, 2, -2]:
                    adj_idx = orig_idx + adj_offset
                    if 0 <= adj_idx < len(lines):
                        adj_line = lines[adj_idx]
                        if _NON_AMOUNT_KEYWORDS.search(adj_line):
                            continue
                        if _SUBTOTAL_LINE.search(adj_line) or _TAX_DISCOUNT_LINE.search(adj_line):
                            continue
                        if _NOT_AMOUNT_LINE.search(adj_line):
                            continue
                        adj_amounts = _extract_amounts_from_line(adj_line)
                        if adj_amounts:
                            return adj_amounts[-1]
                        adj_lenient = _extract_amounts_lenient(adj_line)
                        if adj_lenient:
                            return adj_lenient[-1]

            # For Payment Successful: if we just processed the success line, stop
            if is_payment_success_receipt and _PAYMENT_SUCCESS_RE.search(line):
                break

        return None

    # ── Step 3: multi-pass search in priority order ──────────────────
    # Each pass scans the full receipt in the chosen direction and returns
    # the FIRST match — i.e., the one closest to the bottom (default) or
    # closest to the top (Payment Successful).

    result = _scan_for_amount('strict')
    if result is not None:
        return result

    result = _scan_for_amount('primary')
    if result is not None:
        return result

    result = _scan_for_amount('fallback')
    if result is not None:
        return result

    result = _scan_for_amount('currency')
    if result is not None:
        return result

    # ── Step 4.5: columnar receipt fallback ──────────────────────────
    # If a total keyword EXISTS in the receipt but passes 1-4 couldn't
    # pair it with an amount (labels and values in separate OCR columns),
    # pick the LARGEST standalone monetary amount.
    #
    # Only consider numbers WITH a decimal point (e.g. "5145.00") — plain
    # integers like "15554" are usually bill/ID numbers, not amounts.
    _has_any_total_kw = any(
        (_STRICT_TOTAL_KEYWORDS.search(l) or
         (_TOTAL_KEYWORDS.search(l)
          and not _SUBTOTAL_LINE.search(l)
          and not _TAX_DISCOUNT_LINE.search(l)
          and not _NOT_AMOUNT_LINE.search(l)))
        for l in lines
    )
    if _has_any_total_kw:
        max_amount = None
        for l in lines:
            stripped = l.strip()
            if _NON_AMOUNT_KEYWORDS.search(l):
                continue
            if _SUBTOTAL_LINE.search(l) or _TAX_DISCOUNT_LINE.search(l):
                continue
            if _NOT_AMOUNT_LINE.search(l):
                continue
            # Only standalone numbers WITH a decimal (monetary amounts)
            if (re.match(r"^\d{1,3}(,\d{3})*\.\d{1,2}$", stripped) or
                    re.match(r"^\d{3,8}\.\d{1,2}$", stripped)):
                val = _clean_amount(stripped)
                if val is not None and (max_amount is None or val > max_amount):
                    max_amount = val
        if max_amount is not None:
            return max_amount

    # ── Step 5: standalone numbers (entire line is just a number) ────
    # Scan in the same direction; return the first standalone number found.
    for orig_idx, line in indexed_lines:
        stripped = line.strip()
        if _NON_AMOUNT_KEYWORDS.search(line):
            continue
        if is_payment_success_receipt and _PAYMENT_SUCCESS_RE.search(line):
            break
        if re.match(r"^\d{1,3}(,\d{3})*(\.\d{1,2})?$", stripped) or \
           re.match(r"^\d{3,8}(\.\d{1,2})?$", stripped):
            val = _clean_amount(stripped)
            if val is not None:
                return val

    return None


# Lines that are just a time (HH:MM or HH:MM:SS) — dateutil parses these as
# "today at HH:MM" which produces false date matches.
_TIME_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?$")

# Regex to detect lines that explicitly contain a date label.
# OCR-tolerant: "ate" handles missing "D" in "Date"; no \b before "ate"
# so merged text like "ate06/02/26" is caught.
_DATE_LABEL_RE = re.compile(
    r"d?ate\s*[:.]\s*\d|d?ate\d{1,2}\s*[/\-]|"     # "Date:06/02/26", "ate06/02/26"
    r"\bdate\b|\bdt\b|\bdated\b",                     # standard labels
    re.IGNORECASE,
)

# Explicit date patterns — used to extract dates from merged OCR text
# where dateutil fuzzy might fail (e.g. "ate06/02/26", "19-02-2026 12:00pm").
_EXPLICIT_DATE_RE = re.compile(
    r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})"   # DD/MM/YYYY or DD-MM-YY
    r"|(\d{4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})",    # YYYY-MM-DD
    re.IGNORECASE,
)


# Pattern for "05Feb2026" or "05 Feb 2026" — DDMonYYYY (OCR often merges these)
_DDMONYYYY_RE = re.compile(
    r"(\d{1,2})\s*"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*"
    r"(\d{4})",
    re.IGNORECASE,
)


def _try_parse_date_explicit(line: str):
    """Extract date using explicit regex patterns (handles merged OCR text).

    More robust than dateutil fuzzy for lines like "ate06/02/26" or
    "05Feb2026.06.0333PM" where dateutil can't separate the parts.
    Returns YYYY-MM-DD or None.
    """
    if _NON_AMOUNT_KEYWORDS.search(line):
        return None

    # Try DDMonYYYY first (e.g. "05Feb2026") — must come before numeric
    # pattern so "05Feb2026.06.03" extracts "05 Feb 2026" not "2026.06.03"
    m_mon = _DDMONYYYY_RE.search(line)
    if m_mon:
        try:
            dt = dateutil_parser.parse(
                f"{m_mon.group(1)} {m_mon.group(2)} {m_mon.group(3)}",
                dayfirst=True,
            )
            if 2020 <= dt.year <= 2030:
                return dt.strftime("%Y-%m-%d")
        except (ValueError, OverflowError, TypeError):
            pass

    m = _EXPLICIT_DATE_RE.search(line)
    if not m:
        return None
    try:
        if m.group(1):  # DD/MM/YYYY or DD-MM-YY
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:            # YYYY-MM-DD
            y, mo, d = int(m.group(4)), int(m.group(5)), int(m.group(6))
        # Expand 2-digit year
        if y < 100:
            y += 2000
        if not (2020 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31):
            return None
        dt = dateutil_parser.parse(f"{y}-{mo:02d}-{d:02d}", fuzzy=False)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError, TypeError):
        return None


def _try_parse_date(line: str):
    """Try dateutil fuzzy parse on a single line.  Returns YYYY-MM-DD or None."""
    stripped = line.strip()
    if len(stripped) < 6:
        return None
    if _NON_AMOUNT_KEYWORDS.search(line):
        return None
    if re.match(r"^\d+$", stripped):
        return None
    # Skip time-only lines like "12:04" or "09:08 AM" — dateutil wrongly
    # returns today's date for these.
    if _TIME_ONLY_RE.match(stripped):
        return None
    try:
        dt = dateutil_parser.parse(stripped, fuzzy=True, dayfirst=True)
        if 2020 <= dt.year <= 2030:
            return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError, TypeError):
        pass
    return None


def _parse_date_from_all_lines(lines: list[str]):
    """Extract date from receipt lines using dateutil.parser.

    Passes (in priority order):
    1. Lines with a "date" label → explicit regex extraction (most reliable,
       handles merged OCR like "ate06/02/26").
    2. Explicit date-pattern regex on ALL lines (DD/MM/YY, DD-MM-YYYY, etc.)
       — filters out ID/ref lines but doesn't need a label keyword.
    3. dateutil fuzzy on each line (broad fallback).
    4. YYYYMMDD embedded in Txn ID / Order ID strings.
    """
    # Pass 1: lines with a date label → explicit regex
    for line in lines:
        if _DATE_LABEL_RE.search(line):
            result = _try_parse_date_explicit(line)
            if result:
                return result
            # Also try dateutil fuzzy in case the explicit regex missed it
            result = _try_parse_date(line)
            if result:
                return result

    # Pass 2: explicit date regex on all lines (no label required)
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 6:
            continue
        if _NON_AMOUNT_KEYWORDS.search(line):
            continue
        if re.match(r"^\d+$", stripped):
            continue
        result = _try_parse_date_explicit(line)
        if result:
            return result

    # Pass 3: dateutil fuzzy on each line (skips ID/ref lines & time-only)
    for line in lines:
        result = _try_parse_date(line)
        if result:
            return result

    # Pass 4: YYYYMMDD embedded in transaction/order IDs
    for line in lines:
        m = _TXN_DATE_RE.search(line)
        if m:
            try:
                dt = dateutil_parser.parse(
                    f"{m.group(1)}-{m.group(2)}-{m.group(3)}", fuzzy=False
                )
                if 2020 <= dt.year <= 2030:
                    return dt.strftime("%Y-%m-%d")
            except (ValueError, OverflowError):
                pass

    return None


def _extract_vendor(lines: list[str]) -> str:
    """Extract vendor name — first meaningful line not containing
    'total', 'tax', 'invoice' etc., length > 3.
    Fallback: longest text line.
    """
    for line in lines:
        lower = line.lower().strip()
        if len(lower) <= 3:
            continue
        words = set(lower.split())
        if words & _NON_VENDOR_WORDS:
            continue
        alpha_chars = sum(c.isalpha() for c in line)
        if alpha_chars < len(line) * 0.3:
            continue
        return line.strip()
    # Fallback: longest text line
    if lines:
        return max(lines, key=len).strip()
    return "Unknown"

# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def _ocr_quality_ok(texts: list[str], avg_conf: float) -> bool:
    """Return True if OCR looks like a real receipt, False if garbage."""
    if not texts:
        return False
    all_text = " ".join(texts)
    if not all_text or len(all_text) < 10:
        return False
    # Must contain at least one recognizable receipt pattern:
    # decimal amount, date, currency, or common keyword
    return bool(re.search(
        r"\d+\.\d{2}"                                   # 445.00
        r"|[\₹]|Rs\.?\s*\d|INR\s*\d"                    # ₹, Rs 445
        r"|\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"        # 06/02/26
        r"|\b(?:total|paid|amount|payment|date|tax|gst)\b",
        all_text, re.IGNORECASE,
    ))


def _run_ocr_on_image(img: np.ndarray) -> tuple[list[str], list[float]]:
    """Write image to temp file, run PaddleOCR, return (texts, scores)."""
    tmp_path = "_ocr_tmp.png"
    cv2.imwrite(tmp_path, img)
    del img
    gc.collect()
    try:
        ocr = _get_ocr()
        ocr_result = list(ocr.predict(tmp_path))
    except Exception:
        return [], []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not ocr_result:
        return [], []

    res = ocr_result[0].json.get("res", {})
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])
    del ocr_result
    gc.collect()
    return list(texts), list(scores)


# ---------------------------------------------------------------------------
# LLM-based extraction via Hugging Face VLM — sends the receipt IMAGE
# directly to a vision-language model for accurate parsinggggg
# ---------------------------------------------------------------------------

def _parse_llm_response(text: str) -> dict | None:
    """Parse JSON from LLM response text. Handles markdown fences and extra text."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r'\{[^{}]*\}', cleaned)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    result = {}
    if parsed.get("vendor"):
        v = str(parsed["vendor"]).strip()
        if v.lower() not in ("store name", "store/merchant name", "merchant name", "shop name"):
            result["vendor"] = v
    if parsed.get("amount") is not None:
        try:
            result["amount"] = float(parsed["amount"])
        except (ValueError, TypeError):
            pass
    if parsed.get("date"):
        result["date"] = str(parsed["date"]).strip()
    return result if result else None


def _extract_via_llm(image_source, raw_text: str) -> dict | None:
    """Send receipt to local Ollama for extraction.

    image_source can be a file path (str) or a numpy ndarray.
    Uses VLM (qwen2.5vl:3b) if available — sends the actual image.
    Falls back to text-only (qwen2.5:3b) if VLM fails or isn't installed.
    Fully offline — no data leaves the machine.

    Returns {"vendor": str, "amount": float, "date": "YYYY-MM-DD"} or None.
    """
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:3b")

    # ── Try VLM (vision) first — send actual image ──
    # Resize to max 1024px on longest side before sending — phone photos
    # can be 3000x4000+ which makes VLM inference extremely slow.
    img_b64 = None
    try:
        if isinstance(image_source, np.ndarray):
            img = image_source.copy()
        else:
            img = cv2.imread(image_source)
        if img is not None:
            h, w = img.shape[:2]
            max_side = 1024
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            del img, buf
    except Exception:
        pass

    if img_b64 and "vl" in ollama_model.lower():
        vlm_prompt = (
            "Look at this receipt image carefully. Extract these 3 fields.\n"
            "Return ONLY valid JSON: {\"vendor\": \"...\", \"amount\": number, \"date\": \"YYYY-MM-DD\"}\n\n"
            "1. vendor: The store/restaurant/company name printed at the top of the receipt.\n"
            "2. amount: The FINAL TOTAL amount paid (after discounts/taxes). "
            "Look for 'Grand Total', 'Net Amount', 'Total', 'Amount Paid', 'You Pay'. "
            "Do NOT pick subtotal, tax, CGST, SGST, discount, or MRP.\n"
            "3. date: The billing/transaction date. Indian format DD/MM/YYYY, convert to YYYY-MM-DD.\n\n"
            "Use null for any field you cannot read. No explanation, JSON only."
        )

        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": vlm_prompt,
                    "images": [img_b64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200},
                },
                timeout=180,
            )
            if resp.status_code == 200:
                result = _parse_llm_response(resp.json().get("response", ""))
                if result:
                    print(f"VLM extraction succeeded", file=sys.stderr)
                    return result
            print(f"VLM failed ({resp.status_code}), falling back to text model", file=sys.stderr)
        except Exception as e:
            print(f"VLM error: {e}, falling back to text model", file=sys.stderr)

    # ── Fallback: text-only model with OCR text ──
    if not raw_text or not raw_text.strip():
        return None

    text_model = os.environ.get("OLLAMA_TEXT_MODEL", "qwen2.5:3b")
    raw_lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    total = len(raw_lines)
    top = "\n".join(raw_lines[:7])
    bottom = "\n".join(raw_lines[max(7, total - 15):])

    prompt = (
        "You are parsing an Indian receipt/bill. The OCR text may have spelling errors.\n"
        "I am giving you the TOP (first 7 lines) and BOTTOM (last 15 lines) of the receipt.\n\n"
        "TOP (vendor/store name is here):\n"
        f"{top}\n\n"
        "BOTTOM (total amount and date are here):\n"
        f"{bottom}\n\n"
        "Extract exactly 3 fields. Return ONLY valid JSON, nothing else.\n\n"
        "VENDOR: The business name from the TOP. First or second line. "
        "Ignore addresses, GSTIN, phone numbers. If OCR is garbled, guess the closest real business name.\n\n"
        "AMOUNT: From the BOTTOM, find the FINAL amount paid. "
        "Look for: Grand Total, Net Amount, Net Amt, Total Payable, Amount Paid, You Pay, Balance Due, Total, Amt. "
        "Pick the number next to these keywords. "
        "IGNORE: Sub Total, Item Total, CGST, SGST, Tax, Discount, MRP, Rate, Qty, Volume, Saving.\n\n"
        "DATE: From TOP or BOTTOM, find the transaction date near 'Date', 'Dt', 'Bill Date'. "
        "Indian format DD/MM/YYYY → convert to YYYY-MM-DD. Ignore order IDs and invoice numbers.\n\n"
        '{"vendor": "...", "amount": number, "date": "YYYY-MM-DD"}\n'
        "Use null for fields you cannot find."
    )

    resp = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": text_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200},
        },
        timeout=180,
    )
    if resp.status_code != 200:
        print(f"Ollama error {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return None

    return _parse_llm_response(resp.json().get("response", ""))


def extract_receipt_data(image_path: str) -> dict:
    result = {
        "receipt_file": os.path.basename(image_path),
        "vendor": None,
        "amount": None,
        "date": None,
        "raw_text": "",
        "confidence": 0.0,
        "status": "failed",
    }

    # ── Primary: VLM extracts directly from the receipt image ──
    # No OCR needed — the vision-language model reads the image itself.
    llm = _extract_via_llm(image_path, "")
    if llm:
        print(f"VLM result: {llm}", file=sys.stderr)
        result.update({
            "vendor": llm.get("vendor"),
            "amount": llm.get("amount"),
            "date": llm.get("date"),
            "raw_text": f"[VLM direct extraction — no OCR]\n"
                        f"Vendor: {llm.get('vendor')}\n"
                        f"Amount: {llm.get('amount')}\n"
                        f"Date: {llm.get('date')}",
            "confidence": 0.95,
            "status": "success",
            "llm_raw": llm,
            "regex_vendor": None,
            "regex_amount": None,
            "regex_date": None,
        })
        return result

    # ── Fallback: PaddleOCR + regex if VLM is unavailable ──
    print("VLM unavailable, falling back to OCR + regex", file=sys.stderr)

    try:
        processed_img = preprocess_image(image_path)
    except ValueError:
        return result

    texts, scores = _run_ocr_on_image(processed_img)
    avg_conf = sum(scores) / len(scores) if scores else 0.0

    # Retry WITHOUT EXIF rotation if OCR looks like garbage
    if not _ocr_quality_ok(texts, avg_conf):
        try:
            img_raw = cv2.imread(image_path)
            if img_raw is not None:
                img_raw = _resize_image(img_raw)
                t2, s2 = _run_ocr_on_image(img_raw)
                c2 = sum(s2) / len(s2) if s2 else 0.0
                if c2 > avg_conf or (not texts and t2):
                    texts, scores, avg_conf = t2, s2, c2
        except Exception:
            pass

    if not texts:
        return result

    lines = list(texts)
    full_text = "\n".join(lines)

    extracted_date = _parse_date_from_all_lines(lines)
    extracted_amount = _parse_amount(lines)
    extracted_vendor = _extract_vendor(lines)

    result.update({
        "vendor": extracted_vendor,
        "amount": extracted_amount,
        "date": extracted_date,
        "raw_text": full_text,
        "confidence": round(avg_conf, 4),
        "status": "success",
        "regex_vendor": extracted_vendor,
        "regex_amount": extracted_amount,
        "regex_date": extracted_date,
    })
    return result


def extract_receipt_data_bytes(img_bytes: bytes, filename: str) -> dict:
    """Extract receipt data from raw image bytes (no disk I/O for input)."""
    result = {
        "receipt_file": filename,
        "vendor": None, "amount": None, "date": None,
        "raw_text": "", "confidence": 0.0, "status": "failed",
    }

    # Decode image once — reuse for both VLM and OCR
    try:
        processed_img = preprocess_image_bytes(img_bytes)
    except ValueError:
        return result

    # Primary: VLM extracts directly from the image (numpy array)
    llm = _extract_via_llm(processed_img, "")
    if llm:
        print(f"VLM result: {llm}", file=sys.stderr)
        result.update({
            "vendor": llm.get("vendor"),
            "amount": llm.get("amount"),
            "date": llm.get("date"),
            "raw_text": (f"[VLM direct extraction — no OCR]\n"
                         f"Vendor: {llm.get('vendor')}\n"
                         f"Amount: {llm.get('amount')}\n"
                         f"Date: {llm.get('date')}"),
            "confidence": 0.95, "status": "success",
            "llm_raw": llm,
            "regex_vendor": None, "regex_amount": None, "regex_date": None,
        })
        return result

    # Fallback: PaddleOCR + regex
    print("VLM unavailable, falling back to OCR + regex", file=sys.stderr)
    texts, scores = _run_ocr_on_image(processed_img)
    avg_conf = sum(scores) / len(scores) if scores else 0.0

    # Retry WITHOUT EXIF rotation if OCR looks like garbage
    if not _ocr_quality_ok(texts, avg_conf):
        try:
            arr = np.frombuffer(img_bytes, np.uint8)
            img_raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_raw is not None:
                img_raw = _resize_image(img_raw)
                t2, s2 = _run_ocr_on_image(img_raw)
                c2 = sum(s2) / len(s2) if s2 else 0.0
                if c2 > avg_conf or (not texts and t2):
                    texts, scores, avg_conf = t2, s2, c2
        except Exception:
            pass

    if not texts:
        return result

    lines = list(texts)
    full_text = "\n".join(lines)
    extracted_date = _parse_date_from_all_lines(lines)
    extracted_amount = _parse_amount(lines)
    extracted_vendor = _extract_vendor(lines)

    result.update({
        "vendor": extracted_vendor, "amount": extracted_amount,
        "date": extracted_date, "raw_text": full_text,
        "confidence": round(avg_conf, 4), "status": "success",
        "regex_vendor": extracted_vendor, "regex_amount": extracted_amount,
        "regex_date": extracted_date,
    })
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python ocr_receipt.py <image_path|--stdin> [image_path2 ...]"}))
        sys.exit(1)

    if sys.argv[1] == "--stdin":
        # Binary stdin mode: read length-prefixed files
        # Protocol: 4-byte file count N, then for each file:
        #   4-byte name_len, name_bytes (UTF-8), 4-byte data_len, data_bytes
        import struct
        if sys.platform == "win32":
            import msvcrt
            msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        raw = sys.stdin.buffer

        n_files = struct.unpack(">I", raw.read(4))[0]
        results = []
        for i in range(n_files):
            name_len = struct.unpack(">I", raw.read(4))[0]
            name = raw.read(name_len).decode("utf-8")
            data_len = struct.unpack(">I", raw.read(4))[0]
            data = raw.read(data_len)

            try:
                r = extract_receipt_data_bytes(data, name)
            except Exception as e:
                r = {"receipt_file": name, "vendor": None, "amount": None,
                     "date": None, "raw_text": f"Error: {e}",
                     "confidence": 0.0, "status": "failed"}
            results.append(r)
            print(f"PROGRESS:{len(results)}/{n_files}", file=sys.stderr, flush=True)
        print(json.dumps(results, ensure_ascii=False))
    else:
        # File path mode (backward compatible)
        image_paths = sys.argv[1:]

        if len(image_paths) == 1:
            data = extract_receipt_data(image_paths[0])
            print(json.dumps([data], ensure_ascii=False))
        else:
            results = []
            for path in image_paths:
                try:
                    data = extract_receipt_data(path)
                except Exception as e:
                    data = {
                        "receipt_file": os.path.basename(path),
                        "vendor": None, "amount": None, "date": None,
                        "raw_text": f"Error: {e}",
                        "confidence": 0.0, "status": "failed",
                    }
                results.append(data)
                print(f"PROGRESS:{len(results)}/{len(image_paths)}", file=sys.stderr, flush=True)
            print(json.dumps(results, ensure_ascii=False))
