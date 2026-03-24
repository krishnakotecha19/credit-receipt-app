from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import os
import re
import gc
import json
import subprocess
import sys
import pandas as pd
import requests
from pathlib import Path
from dateutil import parser as dateutil_parser
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image, ImageOps

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


# Path to the subprocess worker scripts (same directory as app.py)
_SCRIPT_DIR = Path(__file__).resolve().parent
_RECEIPT_WORKER = str(_SCRIPT_DIR / "ocr_receipt.py")
_STATEMENT_WORKER = str(_SCRIPT_DIR / "ocr_statement.py")
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOAD_DIR_RECEIPTS = Path("uploads/receipts")
UPLOAD_DIR_STATEMENTS = Path("uploads/statements")
CACHE_DIR_STATEMENTS = Path("cache/statements")
UPLOAD_DIR_RECEIPTS.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR_STATEMENTS.mkdir(parents=True, exist_ok=True)
CACHE_DIR_STATEMENTS.mkdir(parents=True, exist_ok=True)

# Poppler: check multiple known locations
_poppler_candidates = [
    Path(__file__).resolve().parent / "poppler_install" / "poppler-24.08.0" / "Library" / "bin",
    Path(r"C:\poppler-25.12.0\Library\bin"),
    Path(r"C:\poppler\Library\bin"),
]
POPPLER_PATH = None
for _p in _poppler_candidates:
    if _p.exists():
        POPPLER_PATH = str(_p)
        break

MIN_IMAGE_WIDTH = 800

# ---------------------------------------------------------------------------
# Cost Centre Mappings (Entity → Cost Centre Name → GL Code)
# ---------------------------------------------------------------------------
VCARE_MAP = {
    "VCARE: LLP": "8001-626",
    "VCARE: BFT": "3151-583",
    "VCARE: VCE-EDS": "550",
    "VCARE: MGT": "9999",
    "VCARE: Legal": "3999",
    "VCARE: Engineering General": "1998",
    "VCARE: Business Development": "6001",
    "VCARE: FEB General": "3998",
    "VCARE: HR Finance": "9998",
    "VCARE: EDS": "0144",
    "VCARE: SM Shah": "1074",
    "VCARE: Petro Techna (Saudi)": "3106-537",
    "VCARE: Salesforce": "15001",
    "VCARE: Enter Eng": "1114-589",
    "VCARE: Saudi Arabia": "537",
    "VCARE: Chennai Air Water": "3087-512",
    "VCARE: Workshop Mumbai MIDC": "3173-621",
    "VCARE: Air Cargo Mumbai": "3161-601",
    "VCARE: Notus Office General": "1998",
    "VCARE: Shyam Project": "Shyam",
}

SI2TECH_MAP = {
    # Client Specific Domestic
    "Cosmos Impex (India) - Domestic Biz App": "21001",
    "Gulbrandsen Tech (India) - Domestic Biz App": "21002",
    "Gulbrandsen Private Ltd - Domestic Biz App": "21003",
    # Intuitive / Foreign
    "Si2 LLC (Intuitive) - Foreign Biz App": "11001",
    "Si2 LLC (Intuitive) - Foreign IT Infra": "12001",
    "Harshad Shah Financial - Foreign Accounting": "13001",
    # Internal General
    "Internal IT": "99995",
    "Other Operating Costs": "99998",
    "Events / Flames 2025": "99996",
    "New Initiative": "99994",
    "Training & Development": "99997",
    "Management Overhead": "99999",
    "Visa Expense": "99993",
    # Categorized General Codes
    "Foreign - Business App (General)": "11998",
    "Foreign - IT Infra (General)": "12998",
    "Foreign - Accounting/Support (General)": "13998",
    "Foreign - Shared Services": "14998",
    "Foreign - Biz Dev & Marketing": "15998",
    "Domestic - Business App (General)": "21998",
    "Domestic - IT Infra (General)": "22998",
    "Domestic - Accounting/Support (General)": "23998",
    "Domestic - Shared Services": "24998",
    "Domestic - Biz Dev & Marketing (General)": "25998",
}

ENTITY_OPTIONS = ["Si2Tech", "Vcare Global"]
ENTITY_COST_MAP = {"Si2Tech": SI2TECH_MAP, "Vcare Global": VCARE_MAP}
APPROVED_BY_OPTIONS = ["", "Admin"]


def _load_image_fixed(path: Path) -> Image.Image:
    """Load image with correct EXIF orientation (portrait/landscape fix).

    st.image() doesn't apply EXIF rotation — so portrait photos taken on
    phones show sideways.  This applies the rotation and returns a PIL image.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Subprocess wrappers — each OCR engine runs in its own process to avoid
# DLL conflicts between PaddlePaddle (receipts) and PyTorch/DocTR (statements)
# ---------------------------------------------------------------------------

def extract_receipt_data(image_path: str) -> dict:
    """Run VLM receipt extraction in a subprocess (single-receipt mode).
    Falls back to PaddleOCR + regex if VLM is unavailable."""
    fallback = {
        "receipt_file": os.path.basename(image_path),
        "vendor": None, "amount": None, "date": None,
        "raw_text": "", "confidence": 0.0, "status": "failed",
    }
    try:
        proc = subprocess.run(
            [sys.executable, _RECEIPT_WORKER, str(image_path)],
            capture_output=True, text=True, timeout=300,
            cwd=str(_SCRIPT_DIR),
        )
        if proc.returncode != 0:
            fallback["raw_text"] = f"Subprocess error: {proc.stderr[:500]}"
            return fallback
        return json.loads(proc.stdout)
    except Exception as e:
        fallback["raw_text"] = f"Subprocess error: {e}"
        return fallback


def extract_receipts_batch(image_paths: list[str], progress_callback=None) -> list[dict]:
    """Run VLM receipt extraction on ALL receipts in ONE subprocess (batch mode).

    VLM reads images directly — no OCR model loading needed.
    Falls back to PaddleOCR + regex per-receipt if VLM is unavailable.
    Reads PROGRESS:<n>/<total> lines from stderr to drive a progress bar.
    """
    if not image_paths:
        return []

    fallbacks = [
        {
            "receipt_file": os.path.basename(p),
            "vendor": None, "amount": None, "date": None,
            "raw_text": "", "confidence": 0.0, "status": "failed",
        }
        for p in image_paths
    ]

    cmd = [sys.executable, _RECEIPT_WORKER] + [str(p) for p in image_paths]
    timeout = max(300, 90 * len(image_paths))  # 90s per receipt, min 5 min

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(_SCRIPT_DIR),
        )

        # Read stderr for progress updates while process runs
        import threading

        stderr_lines = []
        def _read_stderr():
            for line in proc.stderr:
                stderr_lines.append(line)
                if line.startswith("PROGRESS:") and progress_callback:
                    try:
                        done, total = line.strip().split(":")[1].split("/")
                        progress_callback(int(done), int(total))
                    except (ValueError, IndexError):
                        pass

        t = threading.Thread(target=_read_stderr, daemon=True)
        t.start()

        stdout, _ = proc.communicate(timeout=timeout)
        t.join(timeout=5)

        if proc.returncode != 0:
            err = "".join(stderr_lines)[:500]
            for fb in fallbacks:
                fb["raw_text"] = f"Subprocess error: {err}"
            return fallbacks

        results = json.loads(stdout)
        if isinstance(results, list) and len(results) == len(image_paths):
            return results
        return fallbacks

    except Exception as e:
        for fb in fallbacks:
            fb["raw_text"] = f"Subprocess error: {e}"
        return fallbacks


def process_statement_pdf(pdf_path: str) -> list[dict]:
    """Run DocTR statement OCR in a subprocess."""
    try:
        proc = subprocess.run(
            [sys.executable, _STATEMENT_WORKER, str(pdf_path)] + ([POPPLER_PATH] if POPPLER_PATH else []),
            capture_output=True, text=True, timeout=600,
            cwd=str(_SCRIPT_DIR),
        )
        if proc.returncode != 0:
            return [{"page_number": 1, "raw_ocr_words": [],
                     "status": f"failed: {proc.stderr[:500]}"}]
        return json.loads(proc.stdout)
    except Exception as e:
        return [{"page_number": 1, "raw_ocr_words": [],
                 "status": f"failed: {e}"}]



# ---------------------------------------------------------------------------
# Statement JSON cache (persist parsed statements to disk)
# ---------------------------------------------------------------------------

def _stmt_cache_path(pdf_filename: str) -> Path:
    safe_name = re.sub(r"[^\w\-.]", "_", pdf_filename)
    return CACHE_DIR_STATEMENTS / f"{safe_name}.json"


def _save_stmt_cache(pdf_filename: str, df: pd.DataFrame):
    path = _stmt_cache_path(pdf_filename)
    df.to_json(path, orient="records", date_format="iso", indent=2)


def _load_stmt_cache(pdf_filename: str) -> pd.DataFrame | None:
    path = _stmt_cache_path(pdf_filename)
    if path.exists():
        try:
            df = pd.read_json(path, orient="records")
            expected = {"date", "description", "amount", "type", "confidence"}
            if expected.issubset(set(df.columns)):
                return df
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Session state persistence — survives tab switches / WebSocket reconnects
# but starts fresh on a full app restart (new server process).
# ---------------------------------------------------------------------------
_SESSION_CACHE_DIR = Path("cache/session")
_SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_SESSION_CACHE_FILE = _SESSION_CACHE_DIR / "last_session.json"

# Unique ID for this server process — changes on every `streamlit run` restart.
# Module-level code runs once per process, so this stays constant across reruns.
import time as _time
_SERVER_BOOT_ID = str(os.getpid()) + "_" + str(int(_time.time()))

# Keys we persist: DataFrames are stored as JSON records, scalars as-is
_PERSIST_DF_KEYS = ["df_receipts", "df_statements", "df_matches"]
_PERSIST_SCALAR_KEYS = ["_cached_statement_name", "debug_receipt_ocr",
                        "selected_entity"]


def _save_session_state():
    """Persist critical session state to disk so it survives tab switches."""
    cache = {"_boot_id": _SERVER_BOOT_ID}
    for key in _PERSIST_DF_KEYS:
        df = st.session_state.get(key)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            cache[key] = df.to_json(orient="records", date_format="iso")
    for key in _PERSIST_SCALAR_KEYS:
        val = st.session_state.get(key)
        if val is not None:
            try:
                json.dumps(val)  # test serializability
                cache[key] = val
            except (TypeError, ValueError):
                pass
    if cache:
        _SESSION_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _restore_session_state():
    """Restore persisted session state from disk on tab-switch reconnect.

    Only restores if the cache was written by THIS server process (same boot ID).
    A new process (app restart) gets a fresh session — old cache is deleted.
    """
    if not _SESSION_CACHE_FILE.exists():
        return
    # Only restore if session is empty (fresh connection / tab switch)
    if any(st.session_state.get(k) is not None for k in _PERSIST_DF_KEYS):
        return  # already have data, skip
    try:
        cache = json.loads(_SESSION_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return

    # If boot ID doesn't match, this is a new server process → start fresh
    if cache.get("_boot_id") != _SERVER_BOOT_ID:
        _SESSION_CACHE_FILE.unlink(missing_ok=True)
        return

    for key in _PERSIST_DF_KEYS:
        if key in cache:
            try:
                df = pd.read_json(cache[key], orient="records")
                if not df.empty:
                    st.session_state[key] = df
            except Exception:
                pass
    for key in _PERSIST_SCALAR_KEYS:
        if key in cache:
            st.session_state[key] = cache[key]

    # Backward compat: add cost-centre columns if missing from cached df_matches
    _dfm = st.session_state.get("df_matches")
    if _dfm is not None and not _dfm.empty:
        _default_entity = st.session_state.get("selected_entity", ENTITY_OPTIONS[0])
        for col, default in [("entity", _default_entity), ("cost_centre", ""),
                              ("gl_code", ""), ("approved_by", "")]:
            if col not in _dfm.columns:
                _dfm[col] = default


# Restore on every page load (no-op if session already has data or boot ID differs)
_restore_session_state()


def build_rows(ocr_pages: list[dict]) -> tuple[list[str], list[float]]:
    """Reconstruct table rows from raw OCR word positions using layout analysis.

    Args:
        ocr_pages: Output from process_statement_pdf().

    Returns:
        Tuple of (row_strings, ocr_confidences) where ocr_confidences[i] is the
        average OCR confidence for words in row_strings[i].
    """
    # Header words that ALWAYS indicate a non-transaction row
    _HEADER_ALWAYS = {"statement", "page", "opening", "closing",
                       "description", "offers", "explore", "credit card",
                       "gstin", "hsn"}
    # Words that are header-like ONLY when the row has NO date pattern
    # (e.g. "International Transactions" header vs "AMAZON INTERNATIONAL 5432.00")
    _HEADER_IF_NO_DATE = {"total", "balance", "transaction", "amount",
                           "domestic", "international"}
    # Date patterns: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD MMM YYYY, DD/MM/YY
    _DATE_RE = re.compile(
        r"\d{2}[/\-]\d{2}[/\-]\d{2,4}"       # DD/MM/YYYY or DD-MM-YY
        r"|"
        r"\d{4}[/\-]\d{2}[/\-]\d{2}"          # YYYY-MM-DD
        r"|"
        r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{2,4}"  # 19 Jan 2026
        r"|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{2,4}"  # Jan 19, 2026
        , re.IGNORECASE
    )
    all_rows = []
    all_confs = []

    for page_data in ocr_pages:
        if page_data["status"] != "success":
            continue

        words = page_data["raw_ocr_words"]
        if not words:
            continue

        # Compute centers for each word
        for w in words:
            w["center_y"] = (w["y_min"] + w["y_max"]) / 2.0
            w["center_x"] = (w["x_min"] + w["x_max"]) / 2.0

        # Sort by vertical position
        words.sort(key=lambda w: w["center_y"])

        # Cluster words into rows by Y proximity
        row_threshold = 0.015  # relative to page height (normalized 0-1)
        rows_clustered = []
        current_row = [words[0]]

        for w in words[1:]:
            if abs(w["center_y"] - current_row[-1]["center_y"]) < row_threshold:
                current_row.append(w)
            else:
                rows_clustered.append(current_row)
                current_row = [w]
        rows_clustered.append(current_row)

        # Compute page-level zone boundaries from ALL words on this page
        # so cutoffs are stable across rows (not skewed by per-row word count)
        page_all_cx = sorted(w["center_x"] for w in words)
        page_p15 = page_all_cx[max(0, int(len(page_all_cx) * 0.15))]
        page_p85 = page_all_cx[min(len(page_all_cx) - 1, int(len(page_all_cx) * 0.85))]
        date_cutoff = max(page_p15 + 0.02, 0.15)
        amount_cutoff = min(page_p85 + 0.02, 0.82)

        # DEBUG: capture row-level word layout for first page
        if page_data.get("page_number", 1) == 1:
            _row_layout_debug = st.session_state.setdefault("debug_row_layouts", [])
            _row_layout_debug.append({
                "page": page_data.get("page_number", 1),
                "date_cutoff": round(date_cutoff, 4),
                "amount_cutoff": round(amount_cutoff, 4),
                "p15": round(page_p15, 4),
                "p85": round(page_p85, 4),
                "total_words": len(words),
            })

        # Regex: token looks like a monetary amount (digits, commas, dots, optional +/- prefix)
        _AMOUNT_TOKEN_RE = re.compile(
            r"^[+\-]?\d[\d,]*\.?\d*$"
        )

        # Track debit/credit column X positions (learned from header rows)
        _debit_col_x = None
        _credit_col_x = None

        # DEBUG: capture per-row word layouts (first page only, up to 40 rows)
        _is_first_page = page_data.get("page_number", 1) == 1
        _row_word_debug = st.session_state.setdefault("debug_row_words", [])

        # Sort words within each row by X, then zone and reconstruct
        for row_idx_in_page, row_words in enumerate(rows_clustered):
            row_words.sort(key=lambda w: w["center_x"])

            # Check for header/junk rows — skip them, but first scan
            # for column layout clues (debit/credit column positions)
            row_text_lower = " ".join(w["text"].lower() for w in row_words)
            row_has_date = bool(_DATE_RE.search(row_text_lower))
            is_always_header = any(hw in row_text_lower for hw in _HEADER_ALWAYS)
            is_conditional_header = (not row_has_date and
                                     any(hw in row_text_lower for hw in _HEADER_IF_NO_DATE))
            if is_always_header or is_conditional_header:
                # Look for "cr" / "credit" column header to learn the credit X zone
                for w in row_words:
                    wt = w["text"].lower().strip()
                    if wt in ("cr", "credit", "cr."):
                        _credit_col_x = w["center_x"]
                    elif wt in ("dr", "debit", "dr."):
                        _debit_col_x = w["center_x"]
                if _is_first_page and len(_row_word_debug) < 40:
                    _row_word_debug.append({
                        "row": row_idx_in_page,
                        "status": "SKIPPED (header)",
                        "raw_text": row_text_lower[:80],
                        "words": [{"text": w["text"], "cx": round(w["center_x"], 4)}
                                  for w in row_words],
                    })
                continue

            # Skip rows with too few words (likely junk)
            if len(row_words) < 2:
                continue

            date_parts = []
            desc_parts = []
            amount_parts = []          # list of (cleaned_text, center_x)
            _pending_sign = ""         # holds a standalone '+' or '-' from amount zone
            _word_zones = []           # DEBUG: track zone assignment per word

            for w in row_words:
                cx = w["center_x"]
                text = w["text"]
                if cx < date_cutoff:
                    date_parts.append(text)
                    _word_zones.append({"text": text, "cx": round(cx, 4), "zone": "DATE"})
                elif cx > amount_cutoff:
                    # Fix OCR noise in numeric zone: O->0, l->1
                    cleaned = text.replace("O", "0").replace("o", "0")
                    cleaned = cleaned.replace("l", "1")
                    # Strip leading 'R' (OCR misread of rupee symbol)
                    if cleaned.startswith("R") and len(cleaned) > 1 and cleaned[1:2].isdigit():
                        cleaned = cleaned[1:]

                    # Handle standalone '+' or '-' sign (OCR splits sign from number)
                    if cleaned in ("+", "-"):
                        _pending_sign = cleaned
                        _word_zones.append({"text": text, "cx": round(cx, 4), "zone": "AMOUNT(sign)"})
                        continue

                    # Prepend any pending sign to this token
                    if _pending_sign:
                        cleaned = _pending_sign + cleaned
                        _pending_sign = ""

                    # Only keep tokens that actually look like numbers —
                    # reject stray letters / serial numbers that bled into
                    # the amount zone.
                    if _AMOUNT_TOKEN_RE.match(cleaned):
                        amount_parts.append((cleaned, cx))
                        _word_zones.append({"text": text, "cx": round(cx, 4),
                                            "zone": "AMOUNT", "cleaned": cleaned})
                    else:
                        # Not numeric — push back to description
                        desc_parts.append(text)
                        _word_zones.append({"text": text, "cx": round(cx, 4),
                                            "zone": "AMOUNT→DESC(rejected)"})
                else:
                    # If a pending sign was never consumed, it was noise — discard it
                    _pending_sign = ""
                    desc_parts.append(text)
                    _word_zones.append({"text": text, "cx": round(cx, 4), "zone": "DESC"})

            date_str = " ".join(date_parts).strip()
            desc_str = " ".join(desc_parts).strip()

            # --- Determine the transaction amount and whether it's credit ---
            # Filter to only proper amount tokens (contain comma or dot)
            proper_amounts = [(t, x) for t, x in amount_parts if "," in t or "." in t]
            if not proper_amounts:
                # Fall back to all amount tokens
                proper_amounts = amount_parts

            row_is_credit = False

            if len(proper_amounts) >= 2:
                # Multiple amount tokens — pick the last proper one
                amount_str = proper_amounts[-1][0]
            elif len(proper_amounts) == 1:
                amount_str = proper_amounts[0][0]
            else:
                amount_str = ""

            # --- Credit detection via Geometric Welding ---
            # ocr_statement.py probes a pixel strip left of each amount
            # token for a '+' sign mark that DocTR can't recognise as text.
            # If any amount-zone word has has_plus_prefix=True → credit.
            for w in row_words:
                if w.get("has_plus_prefix") and w["center_x"] > amount_cutoff:
                    row_is_credit = True
                    break

            # If credit detected, prepend '+' so downstream (Qwen + validation) can see it
            if row_is_credit and amount_str and not amount_str.startswith("+"):
                amount_str = "+" + amount_str

            # DEBUG: save row layout for first page
            if _is_first_page and len(_row_word_debug) < 40:
                _row_word_debug.append({
                    "row": row_idx_in_page,
                    "status": "KEPT" if re.search(r"\d{2}[/\-]\d{2}", date_str) else "SKIPPED (no date)",
                    "date_parts": date_str,
                    "desc_parts": desc_str[:50],
                    "amount_parts_raw": " | ".join(t for t, _ in amount_parts),
                    "amount_final": amount_str,
                    "is_credit": row_is_credit,
                    "n_amount_tokens": len(amount_parts),
                    "debit_col_x": round(_debit_col_x, 4) if _debit_col_x else None,
                    "credit_col_x": round(_credit_col_x, 4) if _credit_col_x else None,
                    "words": _word_zones,
                })

            # Clean up date: remove trailing junk digit from OCR (e.g. "19/01/20261" -> "19/01/2026")
            date_match = _DATE_RE.search(date_str)
            if date_match:
                date_str = date_match.group(0)

            # Only keep rows that look like actual transactions (must have a date pattern)
            if not date_match:
                continue

            # Build the row string
            parts = []
            if date_str:
                parts.append(date_str)
            if desc_str:
                parts.append(desc_str)
            if amount_str:
                parts.append(amount_str)

            reconstructed = " | ".join(parts)
            if reconstructed.strip():
                all_rows.append(reconstructed)
                # Average OCR confidence for this row's words
                avg_conf = sum(w["confidence"] for w in row_words) / len(row_words)
                all_confs.append(avg_conf)

    # Merge rows with no amount into the next row's description
    merged_rows = []
    merged_confs = []
    i = 0
    while i < len(all_rows):
        row = all_rows[i]
        conf = all_confs[i]
        # If row has no pipe-separated amount section (no digits after last pipe)
        segments = row.split(" | ")
        last_seg = segments[-1] if segments else ""
        has_amount = bool(re.search(r"\d+\.?\d*", last_seg)) and len(segments) >= 2
        if not has_amount and i + 1 < len(all_rows):
            # Merge with next row's description
            next_segments = all_rows[i + 1].split(" | ")
            if len(next_segments) >= 2:
                next_segments[1] = row.replace(" | ", " ") + " " + next_segments[1]
                all_rows[i + 1] = " | ".join(next_segments)
                # Average the confidences of merged rows
                all_confs[i + 1] = (conf + all_confs[i + 1]) / 2.0
            else:
                merged_rows.append(row)
                merged_confs.append(conf)
            i += 1
        else:
            merged_rows.append(row)
            merged_confs.append(conf)
            i += 1

    return merged_rows, merged_confs


def call_qwen(rows: list[str]) -> list[dict]:
    """Send reconstructed rows to local Qwen2.5-3B (via Ollama) for semantic
    parsing into structured transactions. Fully offline.

    Args:
        rows: Output from build_rows().

    Returns:
        List of dicts with keys: date (str), description (str),
        amount (float), type ('debit' | 'credit').
    """
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

    all_transactions = []
    debug_log = []
    chunk_size = 20

    # Store the raw rows being sent for debugging
    st.session_state["qwen_input_rows"] = rows

    for chunk_start in range(0, len(rows), chunk_size):
        chunk = rows[chunk_start:chunk_start + chunk_size]

        # Build row strings with optional token-level hints for difficult rows
        annotated_rows = []
        for row_str in chunk:
            segments = row_str.split(" | ")
            hints = []
            if len(segments) >= 1:
                hints.append(f"[DATE_ZONE: {segments[0]}]")
            if len(segments) >= 2:
                hints.append(f"[DESC_ZONE: {segments[1]}]")
            if len(segments) >= 3:
                hints.append(f"[AMOUNT_ZONE: {segments[2]}]")
            annotated_rows.append(f"{row_str}  // hints: {' '.join(hints)}")

        rows_text = "\n".join(annotated_rows)

        prompt = (
            "You are an expert financial document parser.\n\n"
            "Convert each row into JSON with fields:\n"
            "- date (YYYY-MM-DD)\n"
            "- description\n"
            "- amount (float)\n"
            "- type (debit/credit/unknown)\n\n"
            "Rules:\n"
            "- Dates must be normalized to YYYY-MM-DD (input is DD/MM/YYYY Indian format)\n"
            "- Remove currency symbols\n"
            "- Use DR → debit, CR → credit\n"
            "- Amounts with '+' prefix are credits; others are debits\n"
            "- If type is unclear, use 'unknown'\n"
            "- Ignore invalid/header rows\n"
            "- Each row has token hints after // showing which zone each part belongs to\n\n"
            "Examples:\n"
            'Input: "12/02/2026 | Amazon Pay | 1,200.00  // hints: [DATE_ZONE: 12/02/2026] [DESC_ZONE: Amazon Pay] [AMOUNT_ZONE: 1,200.00]"\n'
            'Output: {"date":"2026-02-12","description":"Amazon Pay","amount":1200.00,"type":"debit"}\n\n'
            'Input: "13/02/2026 | Swiggy | +450.00  // hints: [DATE_ZONE: 13/02/2026] [DESC_ZONE: Swiggy] [AMOUNT_ZONE: +450.00]"\n'
            'Output: {"date":"2026-02-13","description":"Swiggy","amount":450.00,"type":"credit"}\n\n'
            "Return ONLY a valid JSON array. No explanation.\n\n"
            f"Now process these rows:\n{rows_text}"
        )

        parsed = None
        last_raw = ""
        error_msg = ""

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 2000},
                    },
                    timeout=300,
                )

                if resp.status_code != 200:
                    error_msg = f"Ollama error {resp.status_code}: {resp.text[:200]}"
                    continue

                raw_text = resp.json().get("response", "")
                if not raw_text:
                    error_msg = "Ollama returned empty response"
                    continue

                last_raw = raw_text

                # Strip markdown code fences if present
                cleaned = raw_text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)

                # Extract JSON array
                arr_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
                if arr_match:
                    cleaned = arr_match.group(0)

                parsed = json.loads(cleaned)
                break
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                error_msg = f"JSON parse error (attempt {attempt+1}): {e}\nRaw: {last_raw[:300]}"
                if last_raw:
                    prompt = (
                        "The following output is not valid JSON. "
                        "Fix it and return ONLY a valid JSON array:\n\n"
                        f"{last_raw}"
                    )
                continue
            except requests.RequestException as e:
                error_msg = f"Ollama connection error: {e}"
                break

        if parsed and isinstance(parsed, list):
            for txn in parsed:
                if isinstance(txn, dict) and "description" in txn:
                    try:
                        amt = float(txn.get("amount", 0))
                    except (ValueError, TypeError):
                        amt = 0.0
                    all_transactions.append({
                        "date": txn.get("date", ""),
                        "description": txn.get("description", ""),
                        "amount": amt,
                        "type": txn.get("type", "debit"),
                    })
            debug_log.append({
                "chunk": f"rows {chunk_start+1}-{chunk_start+len(chunk)}",
                "status": "OK",
                "parsed_count": len(parsed),
                "sample": str(parsed[:2])[:200] if parsed else "",
            })
        else:
            debug_log.append({
                "chunk": f"rows {chunk_start+1}-{chunk_start+len(chunk)}",
                "status": "FAILED",
                "error": error_msg[:300],
                "raw_response": last_raw[:300],
            })

    # Store debug log for UI display
    st.session_state["qwen_debug"] = debug_log
    st.session_state["qwen_raw_output"] = all_transactions

    return all_transactions


def validate_and_store_transactions(
    qwen_output: list[dict], ocr_row_confidences: list[float],
    original_rows: list[str] | None = None,
) -> pd.DataFrame:
    """Validate Qwen-parsed transactions and compute hybrid confidence.

    Confidence = OCR confidence (50%) + LLM success (50%).
    LLM success starts at 1.0 and is penalised for invalid fields.
    Discards rows with final score < 0.2.

    Also detects credit transactions by checking for a '+' prefix in the
    amount zone of the original OCR row strings (fallback when Qwen
    doesn't set type to 'credit').

    Returns:
        DataFrame with columns: date, description, amount, type, confidence.
        Also stored in st.session_state['df_statements'].
    """
    # Build credit row lookup from original rows.
    # The '+' prefix is added by build_rows() via geometric welding.
    # We store full row info (date fragment + amount) to match precisely
    # against Qwen output, since index mapping is unreliable (Qwen
    # may skip/merge/reorder rows).
    _credit_rows: list[dict] = []
    if original_rows:
        for row_str in original_rows:
            segments = row_str.split(" | ")
            amount_seg = segments[-1].strip() if len(segments) >= 2 else ""
            if amount_seg.startswith("+"):
                date_seg = segments[0].strip() if segments else ""
                amt_clean = amount_seg.lstrip("+").replace(",", "")
                _credit_rows.append({"date_frag": date_seg, "amount": amt_clean})

    validated = []
    validation_debug = []

    for idx, txn in enumerate(qwen_output):
        # --- LLM success score ---
        llm_success = 1.0
        issues = []

        # Validate date
        date_val = txn.get("date", "")
        try:
            pd.to_datetime(date_val, format="%Y-%m-%d")
        except (ValueError, TypeError):
            llm_success -= 0.4
            issues.append(f"bad date: {date_val!r}")

        # Validate amount
        amount_valid = True
        amt_val = txn.get("amount", 0)
        try:
            amt = float(amt_val)
            if amt <= 0:
                llm_success -= 0.5
                amount_valid = False
                issues.append(f"amount <= 0: {amt_val!r}")
        except (ValueError, TypeError):
            llm_success -= 0.5
            amount_valid = False
            issues.append(f"bad amount: {amt_val!r}")

        # --- OCR confidence (fall back to 0.5 if index out of range) ---
        ocr_conf = ocr_row_confidences[idx] if idx < len(ocr_row_confidences) else 0.5

        # --- Hybrid score ---
        final_score = (ocr_conf * 0.5) + (llm_success * 0.5)

        # Track all rows for debug
        validation_debug.append({
            "idx": idx,
            "date": date_val,
            "desc": txn.get("description", "")[:40],
            "amount": amt_val,
            "type": txn.get("type", ""),
            "llm_success": round(llm_success, 2),
            "ocr_conf": round(ocr_conf, 2),
            "final_score": round(final_score, 2),
            "issues": "; ".join(issues) if issues else "OK",
            "kept": final_score >= 0.2,
        })

        # Discard rows that failed parsing badly
        if final_score < 0.2:
            continue

        # Determine type: if the original row had '+' (from geometric welding),
        # mark as credit. Match by amount + date digits to avoid index misalignment.
        txn_type = txn.get("type", "debit").lower()
        credit_flag = False

        if _credit_rows:
            try:
                txn_amt_str = str(float(txn.get("amount", 0)))
            except (ValueError, TypeError):
                txn_amt_str = ""
            txn_date_str = txn.get("date", "")
            # Extract day+month digits from Qwen date (e.g. "2026-01-27" → "0127")
            txn_date_parts = txn_date_str.replace("-", "")  # "20260127"

            for cr in _credit_rows:
                # Compare amounts (handle float precision: "3598.26" vs "3598.26")
                try:
                    if abs(float(cr["amount"]) - float(txn_amt_str)) > 0.01:
                        continue
                except (ValueError, TypeError):
                    continue
                # Compare dates: original date_frag "27/01/2026" → digits "27012026"
                cr_digits = re.sub(r"[^\d]", "", cr["date_frag"])
                # Check overlap: day+month from original (e.g. "2701") must appear
                # in Qwen date digits (e.g. "20260127")
                if len(cr_digits) >= 4:
                    # Try DD MM from original
                    dd_mm = cr_digits[:2] + cr_digits[2:4]  # "2701"
                    mm_dd = cr_digits[2:4] + cr_digits[:2]  # "0127"
                    if dd_mm in txn_date_parts or mm_dd in txn_date_parts:
                        credit_flag = True
                        break
                else:
                    # Can't verify date — amount alone matched, trust it
                    credit_flag = True
                    break

        overridden = False
        if txn_type != "credit" and credit_flag:
            txn_type = "credit"
            overridden = True

        validation_debug[-1]["credit_flag"] = credit_flag
        validation_debug[-1]["type_final"] = txn_type
        validation_debug[-1]["type_overridden"] = overridden

        validated.append({
            "date": txn.get("date", ""),
            "description": txn.get("description", ""),
            "amount": float(txn.get("amount", 0)) if amount_valid else 0.0,
            "type": txn_type,
            "confidence": round(final_score, 4),
        })

    st.session_state["validation_debug"] = validation_debug

    df = pd.DataFrame(
        validated,
        columns=["date", "description", "amount", "type", "confidence"],
    )

    st.session_state["df_statements"] = df
    return df


_VENDOR_ALIASES = {
    "amazon": ["amzn", "amzn mktp", "amazon.in", "amazon pay"],
    "swiggy": ["swiggy", "bundl technologies"],
    "zomato": ["zomato", "zomato order"],
    "uber": ["uber", "uber trip", "uber eats"],
    "flipkart": ["flipkart", "flipkart pay"],
    "google": ["google", "google play", "google cloud"],
    "paytm": ["paytm", "paytm mall", "one97"],
}


_STOPWORDS = {
    "pvt", "ltd", "limited", "private", "inc", "llc", "corp", "co",
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on",
    "technologies", "services", "solutions", "india", "international",
}


def _remove_stopwords(text: str) -> str:
    """Lowercase, strip, and remove common stopwords."""
    return " ".join(w for w in text.lower().split() if w not in _STOPWORDS).strip()


def _vendor_match(receipt_vendor: str, stmt_desc: str) -> tuple[bool, str]:
    """Check if a receipt vendor matches a statement description.

    Applies lowercase + stopword removal, then tries:
    1. Direct substring match (cleaned text)
    2. Alias lookup (cleaned text)
    3. Token overlap (>=50% of vendor words in description)

    Returns:
        (matched: bool, reason: str) — reason explains why it matched or didn't.
    """
    if not receipt_vendor or not stmt_desc:
        return False, "empty vendor or description"

    rv = _remove_stopwords(receipt_vendor)
    sd = _remove_stopwords(stmt_desc)

    if not rv or not sd:
        return False, f"empty after stopword removal (rv={rv!r}, sd={sd!r})"

    # 1. Direct substring match on cleaned text
    if rv in sd or sd in rv:
        return True, f"substring match: {rv!r} ↔ {sd!r}"

    # 2. Alias lookup (also on cleaned text)
    for _canon, aliases in _VENDOR_ALIASES.items():
        rv_hits = rv in _canon or _canon in rv or any(a in rv for a in aliases)
        sd_hits = sd in _canon or _canon in sd or any(a in sd for a in aliases)
        if rv_hits and sd_hits:
            return True, f"alias match via '{_canon}'"

    # 3. Token overlap: if >=50% of vendor words appear in description
    rv_tokens = set(rv.split())
    sd_tokens = set(sd.split())
    overlap = rv_tokens & sd_tokens
    if rv_tokens and len(overlap) / len(rv_tokens) >= 0.5:
        return True, f"token overlap: {overlap} ({len(overlap)}/{len(rv_tokens)})"

    return False, f"no match (rv={rv!r}, sd={sd!r}, tokens_common={rv_tokens & sd_tokens})"


def match_transactions(
    df_receipts: pd.DataFrame, df_statements: pd.DataFrame
) -> pd.DataFrame:
    """Match receipts against statement debit transactions.

    Only debit transactions are considered for matching (credits are shown
    separately in the Credits tab).

    Tier-based scoring (date + amount are decisive, vendor is supplementary):
        100 — exact date (same day) + exact amount (diff == 0)
         90 — exact amount + date within ±2 days
         70 — vendor match + date within ±2 days + amount within tolerance
         50 — partial match (amount within tolerance only)

    Amount tolerance: exact (diff == 0), close (diff <= ±1), or within ±5%.
    Candidates outside amount tolerance are skipped entirely.

    Returns:
        DataFrame with columns: receipt_file, receipt_vendor, receipt_amount,
        receipt_date, stmt_description, stmt_amount, stmt_date, match_score, status.
    """
    matches = []
    matched_stmt_indices = set()

    # Only match against debit transactions
    debit_mask = df_statements["type"].str.lower() != "credit"
    df_debits = df_statements[debit_mask]

    for r_idx, receipt in df_receipts.iterrows():
        r_vendor = receipt.get("vendor") or ""
        r_amount = receipt.get("amount")
        r_date_str = receipt.get("date")

        # Parse receipt date
        r_date = None
        if r_date_str:
            try:
                r_date = pd.to_datetime(r_date_str)
            except (ValueError, TypeError):
                pass

        best_score = 0
        best_s_idx = None

        for s_idx, stmt in df_debits.iterrows():
            if s_idx in matched_stmt_indices:
                continue

            s_amount = stmt.get("amount")
            s_date_str = stmt.get("date")
            s_desc = stmt.get("description", "")

            # Parse statement date
            s_date = None
            if s_date_str:
                try:
                    s_date = pd.to_datetime(s_date_str)
                except (ValueError, TypeError):
                    pass

            # --- Amount check (mandatory gate) ---
            amount_exact = False
            amount_tol = False
            if r_amount is not None and s_amount is not None:
                try:
                    diff = abs(float(r_amount) - float(s_amount))
                    amount_exact = diff == 0
                    amount_tol = (diff <= 1.0 or
                                  (float(r_amount) > 0 and diff <= float(r_amount) * 0.05))
                except (ValueError, TypeError):
                    pass

            if not amount_tol:
                continue  # amount too far apart, skip

            # --- Date check ---
            date_exact = False
            date_ok = False  # within ±2 days
            if r_date is not None and s_date is not None:
                days_diff = abs((r_date - s_date).days)
                date_exact = days_diff == 0
                date_ok = days_diff <= 2

            # --- Vendor check ---
            vendor_ok, _vendor_reason = _vendor_match(r_vendor, s_desc)

            # --- Tier scoring ---
            score = 0
            if date_exact and amount_exact:
                score = 100
            elif amount_exact and date_ok:
                score = 90
            elif vendor_ok and date_ok and amount_tol:
                score = 70
            elif amount_tol:
                score = 50

            if score > best_score:
                best_score = score
                best_s_idx = s_idx

        # Build match record
        if best_s_idx is not None and best_score > 0:
            matched_stmt_indices.add(best_s_idx)
            stmt_row = df_statements.loc[best_s_idx]
            status = ("auto_approved" if best_score > 90
                      else "review" if best_score >= 50
                      else "unmatched")
            entity = st.session_state.get("selected_entity", ENTITY_OPTIONS[0])
            matches.append({
                "receipt_file": receipt.get("receipt_file", ""),
                "receipt_vendor": r_vendor,
                "receipt_amount": r_amount,
                "receipt_date": r_date_str,
                "stmt_description": stmt_row.get("description", ""),
                "stmt_amount": stmt_row.get("amount"),
                "stmt_date": stmt_row.get("date", ""),
                "match_score": best_score,
                "status": status,
                "entity": entity,
                "cost_centre": "",
                "gl_code": "",
                "approved_by": "",
            })
        else:
            entity = st.session_state.get("selected_entity", ENTITY_OPTIONS[0])
            matches.append({
                "receipt_file": receipt.get("receipt_file", ""),
                "receipt_vendor": r_vendor,
                "receipt_amount": r_amount,
                "receipt_date": r_date_str,
                "stmt_description": "",
                "stmt_amount": None,
                "stmt_date": "",
                "match_score": 0,
                "status": "unmatched",
                "entity": entity,
                "cost_centre": "",
                "gl_code": "",
                "approved_by": "",
            })

    return pd.DataFrame(matches)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Expense Reconciliation", layout="wide")
st.title("AI-Powered Expense Reconciliation")

# Show restored-data indicator if session was restored from disk
_restored_keys = [k for k in _PERSIST_DF_KEYS if st.session_state.get(k) is not None]
if _restored_keys and not st.session_state.get("_session_restore_noted"):
    counts = []
    if st.session_state.get("df_receipts") is not None:
        counts.append(f"{len(st.session_state['df_receipts'])} receipt(s)")
    if st.session_state.get("df_statements") is not None:
        counts.append(f"{len(st.session_state['df_statements'])} transaction(s)")
    if st.session_state.get("df_matches") is not None:
        counts.append(f"{len(st.session_state['df_matches'])} match(es)")
    if counts:
        st.info(f"Previous session restored: {', '.join(counts)}. Upload new files to re-process.")
    st.session_state["_session_restore_noted"] = True

# ---- Sidebar: Upload & Process ----
with st.sidebar:
    st.header("Upload Files")

    # Receipt upload
    st.subheader("📷 Receipts (Images)")
    st.caption("Accepted: JPG, JPEG, PNG — navigate to your **receipts** folder")
    uploaded_receipts = st.file_uploader(
        "Drop receipt images here",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="receipt_uploader",
        help="Select receipt image files (.jpg, .png) from your receipts folder",
    )
    if uploaded_receipts:
        st.success(f"Loaded {len(uploaded_receipts)} receipt(s)")

    st.divider()

    # Statement upload
    st.subheader("📄 Statement (PDF)")
    st.caption("Accepted: PDF only — navigate to your **statements** folder")
    uploaded_statement = st.file_uploader(
        "Drop statement PDF here",
        type=["pdf"],
        key="statement_uploader",
        help="Select a credit-card or bank statement PDF from your statements folder",
    )
    if uploaded_statement:
        st.success(f"Loaded: {uploaded_statement.name}")

    st.divider()

    # Entity selection — determines which cost centre dropdown to show
    st.subheader("🏢 Entity")
    selected_entity = st.selectbox(
        "Select Company",
        options=ENTITY_OPTIONS,
        key="selected_entity",
        help="Cost centre options in the final report will be based on this selection",
    )

    st.divider()

    # Process button
    process_clicked = st.button("Process", type="primary", width="stretch")

# ---- Save uploaded files to disk & process ONLY when Process is clicked ----
receipt_files = []
statement_files = []

if process_clicked:
    # Save files to disk only when processing
    if uploaded_receipts:
        for uf in uploaded_receipts:
            dest = UPLOAD_DIR_RECEIPTS / uf.name
            dest.write_bytes(uf.getbuffer())
            receipt_files.append(dest)

    if uploaded_statement:
        dest = UPLOAD_DIR_STATEMENTS / uploaded_statement.name
        dest.write_bytes(uploaded_statement.getbuffer())
        statement_files.append(dest)

with st.sidebar:
    if uploaded_receipts:
        st.caption(f"{len(uploaded_receipts)} receipt(s) ready")
    if uploaded_statement:
        st.caption(f"1 statement ready")

# ---- Processing ----
if process_clicked:
    # ── STEP 1: Process Receipts ──
    if receipt_files:
        progress = st.sidebar.progress(0, text="Step 1/2 — Processing receipts…")

        def _receipt_progress(done, total):
            progress.progress(done / total, text=f"Receipt {done}/{total}")

        receipt_records = extract_receipts_batch(
            [str(f) for f in receipt_files],
            progress_callback=_receipt_progress,
        )

        # Store per-receipt debug info
        receipt_debug = st.session_state.setdefault("debug_receipt_ocr", {})
        for data in receipt_records:
            receipt_debug[data.get("receipt_file", "")] = {
                "raw_text": data.get("raw_text", ""),
                "vendor": data.get("vendor"),
                "amount": data.get("amount"),
                "date": data.get("date"),
                "confidence": data.get("confidence", 0.0),
                "status": data.get("status", "failed"),
                "llm_raw": data.get("llm_raw"),
                "regex_vendor": data.get("regex_vendor"),
                "regex_amount": data.get("regex_amount"),
                "regex_date": data.get("regex_date"),
            }
        gc.collect()
        progress.empty()

        df_new = pd.DataFrame(receipt_records)
        # Append to existing results if any
        if "df_receipts" in st.session_state:
            existing = st.session_state["df_receipts"]
            # Drop duplicates by receipt_file name
            df_new = pd.concat([existing, df_new]).drop_duplicates(
                subset="receipt_file", keep="last"
            ).reset_index(drop=True)
        st.session_state["df_receipts"] = df_new

        st.sidebar.success(f"Step 1 done — {len(receipt_records)} receipt(s) processed")
        del receipt_records
        gc.collect()
    elif not st.session_state.get("df_receipts") is None:
        st.sidebar.info("No new receipts. Using previously processed data.")
    else:
        st.sidebar.warning("No receipt images found. Upload some first.")

    # ── STEP 2: Process Statements (OCR → Row Reconstruction → Qwen → Validation) ──
    # Cache statement results — check session first, then JSON file on disk.
    _cached_stmt_name = st.session_state.get("_cached_statement_name")
    _current_stmt_name = statement_files[0].name if statement_files else None
    _statement_already_cached = (
        _current_stmt_name
        and _current_stmt_name == _cached_stmt_name
        and st.session_state.get("df_statements") is not None
    )

    # If not in session, try loading from JSON file cache
    if not _statement_already_cached and _current_stmt_name:
        _cached_df = _load_stmt_cache(_current_stmt_name)
        if _cached_df is not None:
            st.session_state["df_statements"] = _cached_df
            st.session_state["_cached_statement_name"] = _current_stmt_name
            _statement_already_cached = True
            st.sidebar.success(
                f"Loaded '{_current_stmt_name}' from cache — "
                f"{len(_cached_df)} transaction(s)"
            )

    if statement_files and not _statement_already_cached:
        # Step 2a: OCR via DocTR
        progress = st.sidebar.progress(0, text="Step 2a/2 — OCR on statements…")
        all_statement_pages = []
        for idx, sfile in enumerate(statement_files):
            try:
                pages = process_statement_pdf(str(sfile))
                all_statement_pages.extend(pages)
            except Exception as e:
                all_statement_pages.append({
                    "page_number": -1,
                    "raw_ocr_words": [],
                    "status": f"failed: {e}",
                })
            gc.collect()
            progress.progress((idx + 1) / len(statement_files),
                              text=f"Statement {idx + 1}/{len(statement_files)}")
        progress.empty()
        st.session_state["statement_pages"] = all_statement_pages

        # Step 2b: Build rows from OCR output
        st.sidebar.info("Step 2b — Reconstructing rows…")
        # Clear debug state from previous runs
        for k in ["debug_row_layouts", "debug_row_words", "debug_credit_rows",
                   "debug_qwen_credits"]:
            st.session_state.pop(k, None)

        # Diagnostic: show page-level OCR stats
        for pg in all_statement_pages:
            pg_num = pg.get("page_number", "?")
            pg_status = pg.get("status", "unknown")
            n_words = len(pg.get("raw_ocr_words", []))
            st.sidebar.caption(f"  Page {pg_num}: {pg_status} — {n_words} words")

        rows, ocr_confidences = build_rows(all_statement_pages)
        del all_statement_pages
        gc.collect()

        st.sidebar.caption(f"  → {len(rows)} row(s) reconstructed")

        # DEBUG: Show rows that contain '+' or 'CR' (potential credits)
        credit_rows_debug = [
            (i, r) for i, r in enumerate(rows)
            if "+" in r.split(" | ")[-1] or "CR" in r.upper().split(" | ")[-1]
        ]
        st.session_state["debug_credit_rows"] = credit_rows_debug
        if credit_rows_debug:
            st.sidebar.caption(f"  → {len(credit_rows_debug)} row(s) with +/CR (potential credits)")
        else:
            st.sidebar.warning("  → 0 rows with +/CR found in OCR output")

        if rows:
            # Step 2c: Send rows to Qwen for parsing
            st.sidebar.info(f"Step 2c — Sending {len(rows)} rows to Qwen…")
            qwen_output = call_qwen(rows)

            # DEBUG: Show what types Qwen returned
            qwen_credits = [
                (i, t.get("description", "")[:30], t.get("amount"), t.get("type"))
                for i, t in enumerate(qwen_output) if t.get("type", "").lower() == "credit"
            ]
            st.session_state["debug_qwen_credits"] = qwen_credits
            st.sidebar.caption(f"  → Qwen returned {len(qwen_credits)} credit(s) out of {len(qwen_output)} txns")

            # Step 2d: Validate and compute hybrid confidence scores
            st.sidebar.info("Step 2d — Validating transactions…")
            df_statements = validate_and_store_transactions(qwen_output, ocr_confidences, rows)

            # DEBUG: Show final credit count in df_statements
            n_credits_final = len(df_statements[df_statements["type"] == "credit"]) if not df_statements.empty else 0
            st.sidebar.caption(f"  → {n_credits_final} credit(s) in final df_statements")

            # Cache the statement name and save to JSON file
            st.session_state["_cached_statement_name"] = _current_stmt_name
            _save_stmt_cache(_current_stmt_name, df_statements)

            st.sidebar.success(
                f"Step 2 done — {len(df_statements)} transaction(s) validated"
            )
            del qwen_output, rows, ocr_confidences
            gc.collect()
        else:
            st.sidebar.warning("No rows reconstructed from statements.")
    elif _statement_already_cached:
        st.sidebar.success(
            f"Statement '{_current_stmt_name}' already processed — "
            f"using cached {len(st.session_state['df_statements'])} transaction(s)"
        )
    elif st.session_state.get("df_statements") is not None:
        st.sidebar.info("No new statements. Using previously processed data.")
    else:
        st.sidebar.info("No statements uploaded. Skipping statement step.")

    # ── STEP 3: Match Receipts ↔ Statements ──
    if "df_receipts" in st.session_state and "df_statements" in st.session_state:
        st.sidebar.info("Step 3 — Matching receipts to transactions…")
        df_matches = match_transactions(
            st.session_state["df_receipts"],
            st.session_state["df_statements"],
        )
        st.session_state["df_matches"] = df_matches
        n_auto = len(df_matches[df_matches["status"] == "auto_approved"])
        n_review = len(df_matches[df_matches["status"] == "review"])
        n_unmatched = len(df_matches[df_matches["match_score"] == 0])

        st.sidebar.success(
            f"Step 3 done — {n_auto} auto-approved, {n_review} review, {n_unmatched} unmatched"
        )

    # Persist session state to disk after all processing steps
    _save_session_state()

# ---------------------------------------------------------------------------
# Helper: generate export bytes
# ---------------------------------------------------------------------------

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helper: editable match table via st.data_editor (scrollable, full-width)
# ---------------------------------------------------------------------------

# Column order for the unified table
_TABLE_COLS = [
    "stmt_description", "stmt_amount", "stmt_date",
    "receipt_vendor", "receipt_amount", "receipt_date",
    "entity", "cost_centre", "gl_code", "approved_by",
    "match_score",
]


def _get_unmatched_stmt_options() -> list[str]:
    """Build dropdown options from statement debits that no receipt is linked to."""
    df_matches = st.session_state.get("df_matches")
    df_stmts = st.session_state.get("df_statements")
    if df_stmts is None or df_stmts.empty:
        return []

    # All debit transactions
    debits = df_stmts[df_stmts["type"].str.lower() != "credit"]

    # Descriptions already linked to a receipt (non-empty stmt_description in matches)
    linked_descs = set()
    if df_matches is not None and not df_matches.empty:
        matched = df_matches[df_matches["status"] != "unmatched"]
        linked_descs = set(matched["stmt_description"].dropna().unique())

    # Build "desc | ₹amt | date" labels for unmatched debits
    options = []
    for _, row in debits.iterrows():
        desc = str(row.get("description", "")).strip()
        if desc and desc not in linked_descs:
            amt = row.get("amount", "")
            dt = row.get("date", "")
            label = f"{desc}  |  ₹{amt}  |  {dt}"
            options.append(label)
    return sorted(set(options))


def _parse_stmt_option(label: str) -> dict:
    """Parse a 'desc | ₹amt | date' label back into stmt fields."""
    parts = [p.strip() for p in label.split("|")]
    desc = parts[0] if len(parts) > 0 else ""
    amt_str = parts[1].replace("₹", "").strip() if len(parts) > 1 else ""
    date_str = parts[2].strip() if len(parts) > 2 else ""
    try:
        amt = float(amt_str)
    except (ValueError, TypeError):
        amt = None
    return {"stmt_description": desc, "stmt_amount": amt, "stmt_date": date_str}


def _get_col_config():
    """Build column config dynamically — Cost Centre options filtered by selected entity."""
    entity = st.session_state.get("selected_entity", ENTITY_OPTIONS[0])
    cc_map = ENTITY_COST_MAP.get(entity, {})
    cc_options = [""] + list(cc_map.keys())

    return {
        "stmt_description": st.column_config.TextColumn("Stmt Description", disabled=True, width="medium"),
        "stmt_amount": st.column_config.NumberColumn("Stmt Amt", disabled=True, format="%.2f", width="small"),
        "stmt_date": st.column_config.TextColumn("Stmt Date", disabled=True, width="small"),
        "receipt_vendor": st.column_config.TextColumn("Receipt Vendor", disabled=True, width="medium"),
        "receipt_amount": st.column_config.NumberColumn("Rcpt Amt", disabled=True, format="%.2f", width="small"),
        "receipt_date": st.column_config.TextColumn("Rcpt Date", disabled=True, width="small"),
        "entity": st.column_config.SelectboxColumn("Entity", options=ENTITY_OPTIONS, width="small", required=True),
        "cost_centre": st.column_config.SelectboxColumn("Cost Centre", options=cc_options, width="large"),
        "gl_code": st.column_config.TextColumn("GL Code", disabled=True, width="small"),
        "approved_by": st.column_config.SelectboxColumn("Approved By", options=APPROVED_BY_OPTIONS, width="small"),
        "match_score": st.column_config.TextColumn("Score", disabled=True, width="small"),
    }


def _sync_edits_back(edited_df: pd.DataFrame, orig_indices):
    """Write edits from data_editor back into session df_matches and auto-fill GL codes."""
    df = st.session_state["df_matches"]
    changed = False
    for pos, orig_idx in enumerate(orig_indices):
        for col in ("entity", "cost_centre", "approved_by"):
            new_val = edited_df.iloc[pos].get(col, "")
            old_val = df.at[orig_idx, col]
            if str(new_val) != str(old_val):
                df.at[orig_idx, col] = new_val if pd.notna(new_val) else ""
                changed = True
        # Auto-fill GL code from entity + cost_centre
        ent = df.at[orig_idx, "entity"]
        cc = df.at[orig_idx, "cost_centre"]
        cc_map = ENTITY_COST_MAP.get(ent, {})
        new_gl = cc_map.get(cc, "")
        if str(df.at[orig_idx, "gl_code"]) != str(new_gl):
            df.at[orig_idx, "gl_code"] = new_gl
            changed = True
    if changed:
        _save_session_state()
        st.rerun()  # Rerun so GL code column refreshes immediately


# ---------------------------------------------------------------------------
# Main area: 5 Tabs
# ---------------------------------------------------------------------------

tab_high, tab_review, tab_unmatched, tab_credits, tab_compare, tab_ocr_debug = st.tabs(
    [
        "Auto-Approved",
        "Needs Review",
        "Unmatched",
        "Credits",
        "Receipt vs Transaction",
        "OCR Debug",
    ]
)

# Retrieve data from session state
_df_matches = st.session_state.get("df_matches")
_df_statements = st.session_state.get("df_statements")
_df_receipts = st.session_state.get("df_receipts")

# ---------------------------------------------------------------------------
# Tab 1: Auto-Approved (score > 90 OR manually reviewed)
# ---------------------------------------------------------------------------
with tab_high:
    st.subheader("Auto-Approved Matches")
    if _df_matches is not None and not _df_matches.empty:
        _high_mask = _df_matches["status"] == "auto_approved"
        df_high = _df_matches[_high_mask]
        if not df_high.empty:
            _hi_orig_idx = list(df_high.index)
            _hi_edit = df_high[_TABLE_COLS].reset_index(drop=True)
            _hi_result = st.data_editor(
                _hi_edit,
                column_config=_get_col_config(),
                use_container_width=True,
                hide_index=True,
                key="de_high",
            )
            _sync_edits_back(_hi_result, _hi_orig_idx)

            st.divider()
            df_high_export = _df_matches[_high_mask][_TABLE_COLS + ["status"]].reset_index(drop=True)
            col_csv, col_xlsx = st.columns(2)
            with col_csv:
                st.download_button(
                    "Download CSV",
                    data=_df_to_csv_bytes(df_high_export),
                    file_name="auto_approved_matches.csv",
                    mime="text/csv",
                )
            with col_xlsx:
                st.download_button(
                    "Download Excel",
                    data=_df_to_excel_bytes(df_high_export),
                    file_name="auto_approved_matches.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.info("No auto-approved matches yet.")
    else:
        st.info("Run processing to see matches here.")

# ---------------------------------------------------------------------------
# Tab 2: Needs Review — table with "Reviewed" button per row
# ---------------------------------------------------------------------------
with tab_review:
    st.subheader("Matches Needing Human Review")
    if _df_matches is not None and not _df_matches.empty:
        _review_mask = _df_matches["status"] == "review"
        df_rev = _df_matches[_review_mask]

        if not df_rev.empty:
            _rv_orig_idx = list(df_rev.index)
            _rv_edit = df_rev[_TABLE_COLS].reset_index(drop=True)
            _rv_edit.insert(0, "approve", False)

            _rv_col_config = {
                "approve": st.column_config.CheckboxColumn("Approve", default=False, width="small"),
                **_get_col_config(),
            }
            _rv_result = st.data_editor(
                _rv_edit,
                column_config=_rv_col_config,
                use_container_width=True,
                hide_index=True,
                key="de_review",
            )
            _sync_edits_back(_rv_result, _rv_orig_idx)

            # Handle bulk approve
            _approved_rows = _rv_result[_rv_result["approve"] == True]
            if not _approved_rows.empty:
                if st.button("Approve Selected", type="primary", key="bulk_approve"):
                    for pos in _approved_rows.index:
                        orig = _rv_orig_idx[pos]
                        st.session_state["df_matches"].at[orig, "status"] = "auto_approved"
                    _save_session_state()
                    st.rerun()

            st.divider()
            df_rev_export = df_rev[_TABLE_COLS + ["status"]].reset_index(drop=True)
            col_csv, col_xlsx = st.columns(2)
            with col_csv:
                st.download_button(
                    "Download CSV",
                    data=_df_to_csv_bytes(df_rev_export),
                    file_name="review_matches.csv",
                    mime="text/csv",
                    key="rev_csv",
                )
            with col_xlsx:
                st.download_button(
                    "Download Excel",
                    data=_df_to_excel_bytes(df_rev_export),
                    file_name="review_matches.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="rev_xlsx",
                )
        else:
            st.info("No matches needing review.")
    else:
        st.info("Run processing to see review items here.")

# ---------------------------------------------------------------------------
# Tab 3: Unmatched
# ---------------------------------------------------------------------------
with tab_unmatched:
    st.subheader("Unmatched Receipts")
    if _df_matches is not None and not _df_matches.empty:
        _un_mask = _df_matches["status"] == "unmatched"
        df_un = _df_matches[_un_mask]
        if not df_un.empty:
            # Build linking options from unmatched statement transactions
            _link_options = ["— Select statement item —"] + _get_unmatched_stmt_options()

            for orig_idx in df_un.index:
                row = _df_matches.loc[orig_idx]
                _vendor = row.get("receipt_vendor", "") or "—"
                _amt = row.get("receipt_amount", "—")
                _date = row.get("receipt_date", "") or "—"

                st.markdown(f"**{_vendor}** — ₹{_amt} — {_date}")

                c1, c2, c3 = st.columns([3, 1.5, 1.5])

                with c1:
                    selected = st.selectbox(
                        "Link to Statement Item",
                        options=_link_options,
                        key=f"link_{orig_idx}",
                        label_visibility="collapsed",
                    )

                # Cost centre & approved by on same row
                _ent = row.get("entity", "") or st.session_state.get("selected_entity", ENTITY_OPTIONS[0])
                _cc_map = ENTITY_COST_MAP.get(_ent, {})
                _cc_opts = [""] + list(_cc_map.keys())
                _cur_cc = row.get("cost_centre", "") or ""
                _cc_idx = _cc_opts.index(_cur_cc) if _cur_cc in _cc_opts else 0

                with c2:
                    new_cc = st.selectbox(
                        "Cost Centre", options=_cc_opts, index=_cc_idx,
                        key=f"un_cc_{orig_idx}", label_visibility="collapsed",
                    )
                    if new_cc != _cur_cc:
                        _df_matches.at[orig_idx, "cost_centre"] = new_cc
                        _df_matches.at[orig_idx, "gl_code"] = _cc_map.get(new_cc, "")
                        _save_session_state()
                        st.rerun()

                _gl = row.get("gl_code", "") or ""
                with c3:
                    st.text_input("GL", value=_gl, disabled=True,
                                  key=f"un_gl_{orig_idx}", label_visibility="collapsed")

                # Handle linking
                if selected != _link_options[0]:
                    stmt_data = _parse_stmt_option(selected)
                    _df_matches.at[orig_idx, "stmt_description"] = stmt_data["stmt_description"]
                    _df_matches.at[orig_idx, "stmt_amount"] = stmt_data["stmt_amount"]
                    _df_matches.at[orig_idx, "stmt_date"] = stmt_data["stmt_date"]
                    _df_matches.at[orig_idx, "match_score"] = "Manual"
                    _df_matches.at[orig_idx, "status"] = "auto_approved"
                    _save_session_state()
                    st.rerun()

                st.divider()

            # Export
            df_un_export = _df_matches[_un_mask][_TABLE_COLS + ["status"]].reset_index(drop=True)
            col_csv, col_xlsx = st.columns(2)
            with col_csv:
                st.download_button(
                    "Download CSV",
                    data=_df_to_csv_bytes(df_un_export),
                    file_name="unmatched_receipts.csv",
                    mime="text/csv",
                    key="un_csv",
                )
            with col_xlsx:
                st.download_button(
                    "Download Excel",
                    data=_df_to_excel_bytes(df_un_export),
                    file_name="unmatched_receipts.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="un_xlsx",
                )

            # Show unmatched receipt images in expander
            with st.expander(f"View {len(df_un)} unmatched receipt image(s)", expanded=False):
                for orig_idx in df_un.index:
                    row = _df_matches.loc[orig_idx]
                    img_path = UPLOAD_DIR_RECEIPTS / row["receipt_file"]
                    col_img, col_data = st.columns([1, 3])
                    with col_img:
                        if img_path.exists():
                            st.image(_load_image_fixed(img_path), caption=row["receipt_file"],
                                     use_container_width=True)
                    with col_data:
                        st.markdown(f"**Vendor:** {row['receipt_vendor'] or '—'}")
                        st.markdown(
                            f"**Amount:** {row['receipt_amount']}"
                            if row["receipt_amount"] else "**Amount:** —"
                        )
                        st.markdown(f"**Date:** {row['receipt_date'] or '—'}")
                    st.divider()
        else:
            st.success("All receipts matched to a transaction.")
    else:
        st.info("Run processing to see unmatched receipts here.")

# ---------------------------------------------------------------------------
# Tab 4: Credits
# ---------------------------------------------------------------------------
with tab_credits:
    st.subheader("Credit / Refund Transactions")
    if _df_statements is not None and not _df_statements.empty:
        df_cr = _df_statements[_df_statements["type"] == "credit"].reset_index(drop=True)
        if not df_cr.empty:
            st.dataframe(df_cr, width="stretch", hide_index=True)
            col_csv, col_xlsx = st.columns(2)
            with col_csv:
                st.download_button(
                    "Download CSV",
                    data=_df_to_csv_bytes(df_cr),
                    file_name="credit_transactions.csv",
                    mime="text/csv",
                    key="cr_csv",
                )
            with col_xlsx:
                st.download_button(
                    "Download Excel",
                    data=_df_to_excel_bytes(df_cr),
                    file_name="credit_transactions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="cr_xlsx",
                )
        else:
            st.info("No credit transactions found in the statement.")
    else:
        st.info("Upload and process a statement to see credits here.")

# ---------------------------------------------------------------------------
# Tab 5: Receipt Image vs Transaction — side-by-side comparison + debug
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("Receipt Image vs Matched Transaction")

    # ── Section A: Show all statement transactions extracted by Qwen ──
    if _df_statements is not None and not _df_statements.empty:
        with st.expander("📄 All Statement Transactions (extracted by Qwen)", expanded=False):
            st.dataframe(_df_statements, width="stretch", hide_index=True)
    elif _df_statements is not None:
        st.info("No transactions extracted from statement.")

    st.markdown("---")

    # ── Section B: Receipt-by-receipt comparison ──
    if _df_matches is not None and not _df_matches.empty and \
       _df_receipts is not None and not _df_receipts.empty:

        for _, match_row in _df_matches.iterrows():
            receipt_file = match_row["receipt_file"]
            img_path = UPLOAD_DIR_RECEIPTS / receipt_file

            score = match_row["match_score"]

            # Colour-code the status
            if str(score) == "Manual":
                badge = ":blue[MANUAL LINK]"
            elif isinstance(score, (int, float)) and score > 90:
                badge = ":green[AUTO-APPROVED]"
            elif isinstance(score, (int, float)) and score >= 50:
                badge = ":orange[NEEDS REVIEW]"
            else:
                badge = ":red[UNMATCHED]"

            st.markdown(f"#### {receipt_file}  —  Score **{score}**  {badge}")

            col_img, col_txn = st.columns([1, 2])

            with col_img:
                if img_path.exists():
                    st.image(_load_image_fixed(img_path), caption=receipt_file,
                             use_container_width=True)
                else:
                    st.warning("Image not found on disk.")
                st.markdown(
                    f"**Vendor:** {match_row['receipt_vendor'] or '—'}  \n"
                    f"**Amount:** {match_row['receipt_amount'] or '—'}  \n"
                    f"**Date:** {match_row['receipt_date'] or '—'}"
                )

                # OCR debug for this receipt
                _receipt_debug = st.session_state.get("debug_receipt_ocr", {})
                _r_debug = _receipt_debug.get(receipt_file)
                if _r_debug:
                    with st.expander("OCR Raw Output", expanded=False):
                        st.caption(f"Status: {_r_debug['status']} | Confidence: {_r_debug['confidence']}")
                        st.text_area(
                            "Raw OCR Text",
                            value=_r_debug.get("raw_text", ""),
                            height=200,
                            key=f"ocr_raw_{receipt_file}",
                            disabled=True,
                        )

            with col_txn:
                if str(score) == "Manual" or (isinstance(score, (int, float)) and score > 0):
                    st.markdown("**Matched Statement Transaction**")
                    st.markdown(
                        f"**Description:** {match_row['stmt_description']}  \n"
                        f"**Amount:** {match_row['stmt_amount']}  \n"
                        f"**Date:** {match_row['stmt_date']}"
                    )

                    # Show scoring breakdown
                    r_date = s_date = None
                    try:
                        r_date = pd.to_datetime(match_row["receipt_date"])
                    except Exception:
                        pass
                    try:
                        s_date = pd.to_datetime(match_row["stmt_date"])
                    except Exception:
                        pass

                    reasons = []
                    if r_date and s_date:
                        day_diff = abs((r_date - s_date).days)
                        exact_tag = " (EXACT)" if day_diff == 0 else ""
                        reasons.append(f"Date diff: **{day_diff} day(s)**{exact_tag}")
                    else:
                        reasons.append("Date diff: **N/A** (could not parse)")

                    r_amt = match_row["receipt_amount"]
                    s_amt = match_row["stmt_amount"]
                    if r_amt is not None and s_amt is not None:
                        try:
                            amt_diff = abs(float(r_amt) - float(s_amt))
                            exact_tag = " (EXACT)" if amt_diff == 0 else ""
                            reasons.append(f"Amount diff: **{amt_diff:.2f}**{exact_tag}")
                        except (ValueError, TypeError):
                            reasons.append("Amount diff: **N/A**")
                    else:
                        reasons.append("Amount diff: **N/A**")

                    vendor_ok, vendor_reason = _vendor_match(
                        match_row["receipt_vendor"] or "",
                        match_row["stmt_description"] or "",
                    )
                    reasons.append(f"Vendor match: **{'Yes' if vendor_ok else 'No'}** — {vendor_reason}")

                    st.markdown("  \n".join(reasons))
                else:
                    st.warning("No matching transaction found for this receipt.")

            # ── Debug expander: show ALL candidates and why each scored or failed ──
            if _df_statements is not None and not _df_statements.empty:
                r_vendor = match_row["receipt_vendor"] or ""
                r_amount = match_row["receipt_amount"]
                r_date_str = match_row["receipt_date"]
                r_date = None
                if r_date_str:
                    try:
                        r_date = pd.to_datetime(r_date_str)
                    except Exception:
                        pass

                with st.expander(f"🔍 Debug: all candidates for {receipt_file}"):
                    debug_rows = []
                    for s_idx, stmt in _df_statements.iterrows():
                        s_amt = stmt.get("amount")
                        s_date_str = stmt.get("date")
                        s_desc = stmt.get("description", "")

                        s_date = None
                        if s_date_str:
                            try:
                                s_date = pd.to_datetime(s_date_str)
                            except Exception:
                                pass

                        # Amount
                        amt_diff = None
                        amt_exact = False
                        amt_tol = False
                        if r_amount is not None and s_amt is not None:
                            try:
                                amt_diff = abs(float(r_amount) - float(s_amt))
                                amt_exact = amt_diff == 0
                                amt_tol = (amt_diff <= 1.0 or
                                           (float(r_amount) > 0 and amt_diff <= float(r_amount) * 0.05))
                            except (ValueError, TypeError):
                                pass

                        # Date
                        day_diff = None
                        date_exact = False
                        date_ok = False
                        if r_date is not None and s_date is not None:
                            day_diff = abs((r_date - s_date).days)
                            date_exact = day_diff == 0
                            date_ok = day_diff <= 2

                        # Vendor
                        v_ok, v_reason = _vendor_match(r_vendor, s_desc)

                        # Score
                        sc = 0
                        score_reason = "—"
                        if not amt_tol:
                            score_reason = "SKIP: amount out of tolerance"
                        elif date_exact and amt_exact:
                            sc = 100
                            score_reason = "exact date + exact amount"
                        elif amt_exact and date_ok:
                            sc = 90
                            score_reason = "exact amount + date ±2d"
                        elif v_ok and date_ok and amt_tol:
                            sc = 70
                            score_reason = "vendor + date + amount"
                        elif amt_tol:
                            sc = 50
                            score_reason = "amount only"

                        debug_rows.append({
                            "stmt_desc": s_desc[:50],
                            "stmt_amt": s_amt,
                            "stmt_date": s_date_str,
                            "amt_diff": round(amt_diff, 2) if amt_diff is not None else "N/A",
                            "amt_ok": "exact" if amt_exact else ("±1" if amt_tol else "NO"),
                            "day_diff": day_diff if day_diff is not None else "N/A",
                            "date_ok": "exact" if date_exact else ("±2d" if date_ok else "NO"),
                            "vendor": "YES" if v_ok else "NO",
                            "vendor_detail": v_reason[:40],
                            "score": sc,
                            "reason": score_reason,
                        })

                    df_debug = pd.DataFrame(debug_rows)
                    # Sort: best scores first
                    df_debug = df_debug.sort_values("score", ascending=False).reset_index(drop=True)
                    st.dataframe(df_debug, width="stretch", hide_index=True)

            st.divider()

        # ── Export ──
        st.markdown("### Export All Data")
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            if _df_receipts is not None and not _df_receipts.empty:
                export_cols = ["receipt_file", "vendor", "amount", "date", "confidence", "status"]
                _df_receipts[export_cols].to_excel(writer, index=False, sheet_name="Receipts")
            if _df_statements is not None and not _df_statements.empty:
                _df_statements.to_excel(writer, index=False, sheet_name="Transactions")
            if _df_matches is not None and not _df_matches.empty:
                _df_matches.to_excel(writer, index=False, sheet_name="Matches")

        col_csv_all, col_xlsx_all, col_stmt = st.columns(3)
        with col_csv_all:
            csv_target = _df_matches if (_df_matches is not None and not _df_matches.empty) else _df_statements
            if csv_target is not None:
                st.download_button(
                    "Download All (CSV)",
                    data=_df_to_csv_bytes(csv_target),
                    file_name="reconciliation_results.csv",
                    mime="text/csv",
                    key="all_csv",
                )
        with col_xlsx_all:
            st.download_button(
                "Download All (Excel)",
                data=buf.getvalue(),
                file_name="reconciliation_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="all_xlsx",
            )
        with col_stmt:
            if _df_statements is not None and not _df_statements.empty:
                st.download_button(
                    "Statement Only (Excel)",
                    data=_df_to_excel_bytes(_df_statements),
                    file_name="parsed_credit_card_statement.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="stmt_xlsx",
                )
    else:
        st.info("Run processing to see receipt vs transaction comparisons here.")

    # ── Pipeline Debug Section — always shown if data exists ──
    st.markdown("---")
    st.markdown("## Pipeline Debug")

    # Debug 1: Raw OCR rows sent to LLM
    _qwen_rows = st.session_state.get("qwen_input_rows")
    if _qwen_rows:
        with st.expander(f"1. OCR Rows sent to LLM ({len(_qwen_rows)} rows)", expanded=False):
            for i, r in enumerate(_qwen_rows):
                st.text(f"  [{i:2d}] {r}")

    # Debug 2: LLM (Ollama) response status
    _qwen_debug = st.session_state.get("qwen_debug")
    if _qwen_debug:
        with st.expander("2. LLM (Qwen2.5-3B) Response Status", expanded=True):
            for entry in _qwen_debug:
                if entry["status"] == "OK":
                    st.success(f"**{entry['chunk']}**: Parsed {entry['parsed_count']} transactions")
                    st.caption(f"Sample: {entry.get('sample', '')}")
                else:
                    st.error(f"**{entry['chunk']}**: FAILED")
                    st.code(entry.get("error", "unknown error"), language=None)
                    if entry.get("raw_response"):
                        st.code(f"Raw response:\n{entry['raw_response']}", language=None)

    # Debug 3: LLM raw output (before validation)
    _qwen_raw = st.session_state.get("qwen_raw_output")
    if _qwen_raw:
        with st.expander(f"3. LLM Raw Output ({len(_qwen_raw)} transactions before validation)", expanded=False):
            df_raw = pd.DataFrame(_qwen_raw)
            st.dataframe(df_raw, width="stretch", hide_index=True)
    elif _qwen_rows:
        st.error("LLM returned 0 transactions. Check that Ollama is running: ollama serve")

    # Debug 4: Validation results (what was kept/rejected and why)
    _val_debug = st.session_state.get("validation_debug")
    if _val_debug:
        with st.expander(f"4. Validation Results ({sum(1 for v in _val_debug if v['kept'])}/{len(_val_debug)} kept)", expanded=True):
            df_val = pd.DataFrame(_val_debug)
            # Highlight rejected rows
            st.dataframe(df_val, width="stretch", hide_index=True)
            rejected = [v for v in _val_debug if not v["kept"]]
            if rejected:
                st.warning(f"{len(rejected)} row(s) rejected:")
                for r in rejected:
                    st.caption(f"  Row {r['idx']}: {r['desc']} — {r['issues']} (score={r['final_score']})")

    # Debug 5: Final validated transactions
    if _df_statements is not None:
        with st.expander(f"5. Final Validated Transactions ({len(_df_statements)} rows)", expanded=False):
            if not _df_statements.empty:
                st.dataframe(_df_statements, width="stretch", hide_index=True)
            else:
                st.error("0 transactions survived validation. Check steps 2-4 above.")

# ---------------------------------------------------------------------------
# Tab 6: OCR Debug — raw OCR text + image for every receipt
# ---------------------------------------------------------------------------
with tab_ocr_debug:
    st.subheader("Receipt Debug")
    st.caption("Shows Qwen VLM extraction results for each receipt image.")

    _receipt_debug_all = st.session_state.get("debug_receipt_ocr", {})

    if _receipt_debug_all:
        # ── Summary table: Qwen VLM JSON output for all receipts ──
        st.markdown("### Qwen VLM Output (all receipts)")
        _vlm_table_rows = []
        for rname, rdbg in _receipt_debug_all.items():
            llm = rdbg.get("llm_raw") or {}
            _vlm_table_rows.append({
                "Receipt": rname,
                "Status": rdbg.get("status", "failed"),
                "VLM Vendor": llm.get("vendor") or rdbg.get("vendor") or "—",
                "VLM Amount": llm.get("amount") or rdbg.get("amount") or "—",
                "VLM Date": llm.get("date") or rdbg.get("date") or "—",
                "Confidence": f"{rdbg.get('confidence', 0.0):.0%}",
                "Source": "VLM" if rdbg.get("llm_raw") else "Regex fallback",
            })
        st.dataframe(pd.DataFrame(_vlm_table_rows), width="stretch", hide_index=True)

        st.markdown("---")

        # ── Per-receipt detail ──
        for receipt_name, dbg in _receipt_debug_all.items():
            img_path = UPLOAD_DIR_RECEIPTS / receipt_name

            status = dbg.get("status", "failed")
            confidence = dbg.get("confidence", 0.0)
            if status == "success" and confidence >= 0.55:
                status_badge = f":green[{status}]"
            elif status == "success":
                status_badge = f":orange[{status} (low confidence)]"
            else:
                status_badge = f":red[{status}]"

            st.markdown(
                f"#### {receipt_name}  —  {status_badge}  "
                f"(confidence: **{confidence:.2%}**)"
            )

            col_img, col_result = st.columns([1, 2])

            with col_img:
                if img_path.exists():
                    st.image(_load_image_fixed(img_path), caption=receipt_name,
                             use_container_width=True)
                else:
                    st.warning("Image not found on disk.")

            with col_result:
                # Show final extracted fields
                st.markdown("**Extracted Fields:**")
                st.markdown(
                    f"**Vendor:** {dbg.get('vendor') or '—'}  \n"
                    f"**Amount:** {dbg.get('amount') or '—'}  \n"
                    f"**Date:** {dbg.get('date') or '—'}"
                )

                # Show VLM JSON response
                llm_raw = dbg.get("llm_raw")
                st.markdown("---")
                st.markdown("**Qwen VLM JSON response:**")
                if llm_raw:
                    st.json(llm_raw)
                else:
                    st.caption("VLM returned nothing — used regex fallback")

                # Show regex fallback if it was used
                if not llm_raw:
                    st.markdown("**Regex fallback:**")
                    st.markdown(
                        f"Vendor: `{dbg.get('regex_vendor') or '—'}`  \n"
                        f"Amount: `{dbg.get('regex_amount') or '—'}`  \n"
                        f"Date: `{dbg.get('regex_date') or '—'}`"
                    )

                # Show raw OCR text only if OCR was used (not VLM)
                raw_text = dbg.get("raw_text", "")
                is_vlm = raw_text.startswith("[VLM direct")
                if not is_vlm and raw_text:
                    with st.expander("Raw OCR Text (fallback)", expanded=False):
                        st.text_area(
                            "OCR Output",
                            value=raw_text,
                            height=300,
                            key=f"ocr_debug_{receipt_name}",
                            disabled=True,
                        )

            st.divider()
    else:
        st.info("Run processing to see receipt debug output.")
