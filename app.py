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
                fb["raw_text"] = f"Subprocess error (exit {proc.returncode}): {err}"
            return fallbacks

        if not stdout or not stdout.strip():
            err = "".join(stderr_lines)[:500]
            for fb in fallbacks:
                fb["raw_text"] = f"Subprocess returned empty stdout. stderr: {err}"
            return fallbacks

        try:
            results = json.loads(stdout)
        except json.JSONDecodeError as e:
            for fb in fallbacks:
                fb["raw_text"] = f"JSON parse error: {e}\nstdout: {stdout[:300]}"
            return fallbacks

        if isinstance(results, list) and len(results) == len(image_paths):
            return results
        for fb in fallbacks:
            fb["raw_text"] = f"Unexpected result: got {type(results).__name__} len={len(results) if isinstance(results, list) else '?'}, expected list of {len(image_paths)}"
        return fallbacks

    except subprocess.TimeoutExpired:
        proc.kill()
        err = "".join(stderr_lines)[:500]
        for fb in fallbacks:
            fb["raw_text"] = f"Subprocess timed out after {timeout}s. stderr: {err}"
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


# Pre-compiled regexes used in build_rows / parse_rows_fast
_CANON_AMT_RE = re.compile(r"\d[\d,]*\.\d{2}")   # canonical decimal amount extractor
_LEADING_IDX_RE = re.compile(r"^\d{1,3}\s+")       # leading row-index digit(s) e.g. "1 "
_HTTPS_RE = re.compile(r"\s*HTTPS?://\S*", re.IGNORECASE)  # URL noise
_TRAILING_R_RE = re.compile(r"\s+R\s*$")            # trailing " R" OCR artifact


def build_rows(ocr_pages: list[dict]) -> tuple[list[str], list[float]]:
    """Reconstruct table rows from raw OCR words with column structure.

    1. Cluster words into rows by Y proximity (adaptive threshold)
    2. Sort each row's words left-to-right by X
    3. Detect column breaks using X gaps between consecutive words
    4. Output each row as "col1 | col2 | col3 | ..." with columns
       separated by pipe, so the LLM can see the table structure
    5. Skip obvious header/junk rows

    Returns:
        Tuple of (row_strings, ocr_confidences).
    """
    _HEADER_WORDS = {"statement", "page", "opening", "closing",
                     "description", "offers", "explore", "credit card",
                     "gstin", "hsn", "rewards", "unbilled"}
    _DATE_RE = re.compile(
        r"\d{2}[/\-]\d{2}[/\-]\d{2,4}"
        r"|\d{4}[/\-]\d{2}[/\-]\d{2}"
        r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{2,4}"
        , re.IGNORECASE
    )

    all_rows = []
    all_confs = []

    for page_data in ocr_pages:
        if page_data.get("status") != "success":
            continue

        words = page_data.get("raw_ocr_words", [])
        if not words:
            continue

        # Compute centers
        for w in words:
            w["center_y"] = (w["y_min"] + w["y_max"]) / 2.0
            w["center_x"] = (w["x_min"] + w["x_max"]) / 2.0

        # Sort by Y
        words.sort(key=lambda w: w["center_y"])

        # Adaptive row threshold
        y_centers = sorted(set(round(w["center_y"], 4) for w in words))
        if len(y_centers) > 3:
            gaps = sorted(
                y_centers[i + 1] - y_centers[i]
                for i in range(len(y_centers) - 1)
                if y_centers[i + 1] - y_centers[i] > 0.002
            )
            if gaps:
                p40 = gaps[int(len(gaps) * 0.4)]
                row_threshold = max(0.005, min(p40 * 0.7, 0.015))
            else:
                row_threshold = 0.008
        else:
            row_threshold = 0.008

        # Cluster into rows
        rows_clustered = []
        current_row = [words[0]]
        for w in words[1:]:
            if abs(w["center_y"] - current_row[-1]["center_y"]) < row_threshold:
                current_row.append(w)
            else:
                rows_clustered.append(current_row)
                current_row = [w]
        rows_clustered.append(current_row)

        # Compute median word gap on this page to detect column breaks
        all_x_gaps = []
        for row_words in rows_clustered:
            row_words.sort(key=lambda w: w["center_x"])
            for j in range(1, len(row_words)):
                gap = row_words[j]["x_min"] - row_words[j - 1]["x_max"]
                if gap > 0:
                    all_x_gaps.append(gap)
        if all_x_gaps:
            all_x_gaps.sort()
            median_gap = all_x_gaps[len(all_x_gaps) // 2]
            col_gap_threshold = max(median_gap * 3, 0.03)
        else:
            col_gap_threshold = 0.05

        # Process each row — split into columns at big X gaps
        for row_words in rows_clustered:
            row_words.sort(key=lambda w: w["center_x"])

            # Skip too few words
            if len(row_words) < 2:
                continue

            # Build columns by detecting X gaps
            columns = []
            current_col_words = [row_words[0]]
            for j in range(1, len(row_words)):
                gap = row_words[j]["x_min"] - row_words[j - 1]["x_max"]
                if gap > col_gap_threshold:
                    columns.append(current_col_words)
                    current_col_words = [row_words[j]]
                else:
                    current_col_words.append(row_words[j])
            columns.append(current_col_words)

            # --- Identify the rightmost amount column ---
            # The amount is always the rightmost column that contains a
            # number with a decimal point.  Check that column's words for
            # '+' (geometric welding or standalone token) to detect credit.
            _AMT_COL_RE = re.compile(r"\d[\d,]*\.\d")

            amount_col_idx = -1
            for ci in range(len(columns) - 1, -1, -1):
                col_text = " ".join(w["text"] for w in columns[ci])
                if _AMT_COL_RE.search(col_text):
                    amount_col_idx = ci
                    break

            # Build column texts
            col_texts = []
            is_credit = False

            for ci, col_words in enumerate(columns):
                if ci == amount_col_idx:
                    # This is the amount column — check for credit
                    # 1) geometric welding / DocTR '+' prefix on any word in this column
                    if any(w.get("has_plus_prefix") for w in col_words):
                        is_credit = True
                    # 2) standalone '+' token in this column
                    if any(w["text"].strip() == "+" for w in col_words):
                        is_credit = True
                    # 3) '+' token in the column just before amount
                    if ci > 0:
                        prev_col_text = " ".join(w["text"] for w in columns[ci - 1]).strip()
                        if prev_col_text == "+":
                            is_credit = True
                            # Don't include the standalone '+' column in output
                            if col_texts and col_texts[-1].strip() == "+":
                                col_texts.pop()

                    # Build amount text: strip standalone +/- from words,
                    # keep only the numeric part
                    amt_parts = []
                    for w in col_words:
                        t = w["text"].strip()
                        # 4) '+' prefix directly inside a word value (e.g. "+7,627.00")
                        if t.startswith("+") and re.match(r"^\+\d", t):
                            is_credit = True
                            t = t[1:]  # strip the sign
                        if t in ("+", "-"):
                            continue  # consumed for credit detection
                        amt_parts.append(t)
                    raw_amt_text = " ".join(amt_parts)

                    # --- Bug 1 fix: re-extract canonical decimal amount ---
                    # When DocTR splits "7,627" and "00" as separate tokens,
                    # joining gives "7,627 00" which LLM or regex may read as
                    # 762700. Re-extract the last \d[\d,]*.\d{2} match instead.
                    canon_matches = _CANON_AMT_RE.findall(raw_amt_text)
                    amt_text = canon_matches[-1] if canon_matches else raw_amt_text

                    # Prepend + if credit so the LLM sees it clearly
                    if is_credit:
                        col_texts.append("+" + amt_text)
                    else:
                        col_texts.append(amt_text)
                else:
                    col_texts.append(" ".join(w["text"] for w in col_words))

            row_text = " | ".join(col_texts)
            row_lower = row_text.lower()

            # Skip headers
            if any(hw in row_lower for hw in _HEADER_WORDS):
                continue

            # Must contain a date pattern
            if not _DATE_RE.search(row_text):
                continue

            # --- Bug 2 fix: clean up description columns (non-amount, non-date) ---
            # Remove leading row-index digit(s) (e.g. "1 "), trailing " R" OCR
            # artifacts, and HTTPS URL noise absorbed from statement watermarks.
            cleaned_cols = row_text.split(" | ")
            if len(cleaned_cols) >= 3:
                # Only clean the middle (description) columns, not date or amount
                for mid_ci in range(1, len(cleaned_cols) - 1):
                    desc = cleaned_cols[mid_ci]
                    desc = _LEADING_IDX_RE.sub("", desc, count=1)
                    desc = _HTTPS_RE.sub("", desc)
                    desc = _TRAILING_R_RE.sub("", desc)
                    cleaned_cols[mid_ci] = desc.strip()
                row_text = " | ".join(cleaned_cols)

            all_rows.append(row_text)
            avg_conf = sum(w["confidence"] for w in row_words) / len(row_words)
            all_confs.append(avg_conf)

    return all_rows, all_confs


def _normalize_date(date_str: str) -> str:
    """Convert DD/MM/YYYY, DD-MM-YYYY, DD/MM/YY etc. to YYYY-MM-DD.
    Returns empty string if unparseable."""
    date_str = date_str.strip()
    if not date_str:
        return ""

    # DD/MM/YYYY or DD-MM-YYYY or DD/MM/YY
    m = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})$", date_str)
    if m:
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if yy < 100:
            yy += 2000
        if 1 <= mm <= 12 and 1 <= dd <= 31 and 2020 <= yy <= 2030:
            return f"{yy:04d}-{mm:02d}-{dd:02d}"
        # Maybe MM/DD/YYYY — try swapping if dd > 12
        if dd > 12 and 1 <= mm <= 31 and 1 <= dd <= 12:
            return f"{yy:04d}-{dd:02d}-{mm:02d}"
        return ""

    # YYYY-MM-DD (already correct)
    m = re.match(r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})$", date_str)
    if m:
        yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 2020 <= yy <= 2030 and 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yy:04d}-{mm:02d}-{dd:02d}"
        return ""

    # DD MMM YYYY / MMM DD, YYYY (e.g. "19 Jan 2026", "Jan 19, 2026")
    try:
        dt = dateutil_parser.parse(date_str, dayfirst=True)
        if 2020 <= dt.year <= 2030:
            return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError, TypeError):
        pass

    return ""


def parse_rows_with_llm(rows: list[str]) -> list[dict]:
    """Send column-structured OCR rows to local Qwen 3B for parsing.

    Input rows look like: "25/02/2026 | 15:40 ING*MAKE MY TRIP | 7,627.00"
    Columns are separated by | based on X-gap detection in build_rows.
    [CREDIT] prefix means the amount has a + sign.

    The LLM sees the table structure and extracts date, description, amount.
    Sent in chunks of 15 rows to keep context small for 3B model.

    Returns:
        List of dicts: date, description, amount, type.
    """
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    # Bug 4 fix: use the TEXT model (qwen2.5:3b), not the VISION model (qwen2.5vl:3b).
    # Statement rows are plain text — sending them to the VLM causes hallucinations.
    ollama_model = os.environ.get("OLLAMA_TEXT_MODEL", "qwen2.5:3b")

    all_transactions = []
    debug_log = []
    chunk_size = 15  # 3B model has limited context; 15 rows is safer than 30

    st.session_state["qwen_input_rows"] = rows

    for chunk_start in range(0, len(rows), chunk_size):
        chunk = rows[chunk_start:chunk_start + chunk_size]
        rows_text = "\n".join(f"ROW{i+1}: {r}" for i, r in enumerate(chunk))

        prompt = (
            "Parse these credit card statement rows into JSON.\n"
            "Each row format: DATE | DESCRIPTION | AMOUNT\n"
            "Columns are separated by |\n\n"
            "RULES:\n"
            "- date: Convert DD/MM/YYYY to YYYY-MM-DD format.\n"
            "- description: Use only the MERCHANT/VENDOR NAME. "
            "Remove: leading single digits (like '1'), HH:MM timestamps, "
            "Ref# IDs, HTTPS URLs, trailing 'R' characters. Keep the merchant name intact.\n"
            "- amount: The rightmost column (after the last |) is the INR amount. "
            "If the amount starts with + it is type=credit, otherwise type=debit. "
            "Remove commas and the + sign. Return as a plain float (e.g. 7627.00).\n"
            "- INCLUDE ALL rows, including fees, taxes, and payments.\n"
            "- If a field is unclear, make your best guess — do NOT skip the row.\n\n"
            "Examples:\n"
            'ROW1: 25/02/2026 | 14:06 MAKE MY TRIP INDIA PVT L | 7,627.00\n'
            'ROW2: 27/01/2026 | 00:00 IRCTC MPP NEW DELHI | +3,598.26\n'
            'ROW3: 13/02/2026 | 1 10:48 ING*MAKE MY TRIP INDI | 17,504.00\n'
            'ROW4: 10/03/2026 | 21:42 AKFOODSVADODARA | 530.00\n\n'
            'Output: [\n'
            '  {"date":"2026-02-25","description":"MAKE MY TRIP INDIA PVT L","amount":7627.00,"type":"debit"},\n'
            '  {"date":"2026-01-27","description":"IRCTC MPP NEW DELHI","amount":3598.26,"type":"credit"},\n'
            '  {"date":"2026-02-13","description":"MAKE MY TRIP INDI","amount":17504.00,"type":"debit"},\n'
            '  {"date":"2026-03-10","description":"AKFOODSVADODARA","amount":530.00,"type":"debit"}\n'
            ']\n\n'
            "Return ONLY a valid JSON array with exactly one object per ROW. No explanations.\n\n"
            f"{rows_text}"
        )

        parsed = None
        raw_text = ""

        for attempt in range(3):  # up to 3 attempts per chunk
            try:
                resp = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 2000, "num_ctx": 8192},
                    },
                    timeout=120,
                )
                if resp.status_code != 200:
                    continue

                raw_text = resp.json().get("response", "")
                if not raw_text:
                    continue

                cleaned = raw_text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
                arr_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
                if arr_match:
                    cleaned = arr_match.group(0)

                parsed = json.loads(cleaned)
                break
            except json.JSONDecodeError:
                # Try to recover partial JSON on last attempt
                if attempt == 2 and raw_text:
                    # Extract individual JSON objects and reassemble
                    obj_matches = re.findall(r'\{[^{}]+\}', raw_text, re.DOTALL)
                    if obj_matches:
                        try:
                            parsed = [json.loads(o) for o in obj_matches]
                            break
                        except (json.JSONDecodeError, ValueError):
                            pass
                continue
            except requests.RequestException as e:
                debug_log.append({
                    "row": f"Chunk {chunk_start+1}-{chunk_start+len(chunk)}",
                    "status": f"CONNECTION ERROR: {e}"
                })
                break

        if parsed and isinstance(parsed, list):
            for txn in parsed:
                if isinstance(txn, dict) and txn.get("amount"):
                    try:
                        amt = round(float(txn.get("amount", 0)), 2)
                    except (ValueError, TypeError):
                        amt = 0.0
                    if amt > 0:
                        all_transactions.append({
                            "date": txn.get("date", ""),
                            "description": txn.get("description", ""),
                            "amount": amt,
                            "type": txn.get("type", "debit"),
                        })
            debug_log.append({
                "row": f"Chunk {chunk_start+1}-{chunk_start+len(chunk)}",
                "status": f"OK ({len(parsed)} parsed, {len(all_transactions)} total kept)"
            })
        else:
            # LLM failed all 3 attempts — fall back to regex for this chunk
            debug_log.append({
                "row": f"Chunk {chunk_start+1}-{chunk_start+len(chunk)}",
                "status": f"LLM FAILED after 3 attempts, using regex fallback. Raw: {raw_text[:100]}"
            })
            fallback_txns = parse_rows_fast(chunk)
            all_transactions.extend(fallback_txns)

    st.session_state["qwen_debug"] = debug_log
    st.session_state["qwen_raw_output"] = all_transactions
    return all_transactions



def parse_rows_columnar(rows: list[str]) -> list[dict]:
    """PRIMARY parser: directly use the pipe-column structure from build_rows().

    build_rows() already did the spatial work — it separated words into columns
    using DocTR bounding boxes and joined them as:
        'DD/MM/YYYY | <time> MERCHANT NAME | [+]1,234.56'

    We trust that structure and simply:
      - segment[0]  → date
      - segment[-1] → amount (+ prefix = credit)
      - segment[1:-1] joined → description (cleaned)

    No LLM, no regex guessing on the full row. Deterministic, instant, never skips.
    """
    _AMOUNT_RE = re.compile(r'[+\-]?\d[\d,]*\.\d{2}')
    _TIME_PREFIX_RE = re.compile(r'^\d{1,2}:\d{2}\s*')
    _LEADING_IDX_RE2 = re.compile(r'^\d{1,3}\s+')

    transactions = []
    debug_log = []
    st.session_state["qwen_input_rows"] = rows

    for row in rows:
        segments = [s.strip() for s in row.split(' | ')]
        if len(segments) < 2:
            debug_log.append({"row": row[:80], "status": "SKIP (too few columns)"})
            continue

        # ── Date: always the first segment ──
        date_raw = segments[0].split()[0] if segments[0] else ""
        date_norm = _normalize_date(date_raw)
        if not date_norm:
            debug_log.append({"row": row[:80], "status": f"SKIP (bad date: {date_raw!r})"})
            continue

        # ── Amount: always the last segment ──
        amt_seg = segments[-1].strip()
        is_credit = amt_seg.startswith('+')
        amt_match = _AMOUNT_RE.search(amt_seg)
        if not amt_match:
            debug_log.append({"row": row[:80], "status": "SKIP (no amount in last col)"})
            continue
        try:
            amount_val = round(float(amt_match.group(0).lstrip('+-').replace(',', '')), 2)
        except ValueError:
            debug_log.append({"row": row[:80], "status": "SKIP (amount parse error)"})
            continue
        if amount_val <= 0:
            debug_log.append({"row": row[:80], "status": "SKIP (amount <= 0)"})
            continue

        # ── Description: everything in between, cleaned ──
        if len(segments) >= 3:
            desc = ' '.join(segments[1:-1])
        else:
            # Only 2 columns — description may be embedded in date column after the date
            desc = ' '.join(segments[0].split()[1:])  # words after date token

        # Clean up description: strip leading index digit, timestamp, URL, trailing R
        desc = _LEADING_IDX_RE2.sub('', desc, count=1)
        desc = _TIME_PREFIX_RE.sub('', desc)
        desc = _HTTPS_RE.sub('', desc)
        desc = _TRAILING_R_RE.sub('', desc)
        desc = desc.strip(' |-')

        transactions.append({
            "date": date_norm,
            "description": desc,
            "amount": amount_val,
            "type": "credit" if is_credit else "debit",
        })
        debug_log.append({"row": row[:80], "status": "OK"})

    st.session_state["qwen_debug"] = debug_log
    st.session_state["qwen_raw_output"] = transactions
    return transactions


def parse_rows_fast(rows: list[str]) -> list[dict]:
    """Regex fallback parser — used when Ollama is not available.

    Extracts date (first date pattern), amount (last decimal number),
    description (everything in between) from column-structured rows.
    """
    _DATE_RE = re.compile(
        r"\d{2}[/\-]\d{2}[/\-]\d{2,4}"
        r"|\d{4}[/\-]\d{2}[/\-]\d{2}"
        , re.IGNORECASE
    )
    _AMOUNT_RE = re.compile(r"[+\-]?\d[\d,]*\.\d{1,2}")
    # ₹-prefixed amount: optional + before ₹, then the number
    _RUPEE_AMOUNT_RE = re.compile(r"(\+)?\s*[₹]\s*(\d[\d,]*\.\d{1,2})")
    # Foreign currency codes to skip
    _FOREIGN_CUR_RE = re.compile(r"\b(THB|USD|EUR|GBP|AED|SGD|JPY)\s+\d", re.IGNORECASE)

    transactions = []
    debug_log = []
    st.session_state["qwen_input_rows"] = rows

    for row_str in rows:
        text = row_str.strip()
        # Credit: the rightmost amount column starts with +
        # build_rows already prepends + to the amount column
        is_credit = False

        date_match = _DATE_RE.search(text)
        if not date_match:
            debug_log.append({"row": row_str[:80], "status": "SKIP (no date)"})
            continue

        date_normalized = _normalize_date(date_match.group(0))

        # Priority 1: look for ₹-prefixed amount (the INR amount we want)
        rupee_match = _RUPEE_AMOUNT_RE.search(text)
        if rupee_match:
            if rupee_match.group(1) == "+":
                is_credit = True
            amount_clean = rupee_match.group(2).replace(",", "")
            try:
                amount_val = round(float(amount_clean), 2)
            except ValueError:
                amount_val = 0.0
            desc = text[date_match.end():rupee_match.start()].strip()
        else:
            # Priority 2: last decimal number, but skip foreign currency amounts
            amounts = list(_AMOUNT_RE.finditer(text))
            if not amounts:
                debug_log.append({"row": row_str[:80], "status": "SKIP (no amount)"})
                continue

            # Filter out amounts that are preceded by foreign currency codes
            inr_amounts = []
            for m in amounts:
                before_text = text[max(0, m.start() - 5):m.start()]
                if not _FOREIGN_CUR_RE.search(before_text):
                    inr_amounts.append(m)

            # Use filtered INR amounts if any, otherwise fall back to all
            chosen_amounts = inr_amounts if inr_amounts else amounts
            last_amt = chosen_amounts[-1]

            amount_clean = last_amt.group(0).lstrip("+-").replace(",", "")
            try:
                amount_val = round(float(amount_clean), 2)
            except ValueError:
                continue
            desc = text[date_match.end():last_amt.start()].strip()

            if not is_credit and last_amt.group(0).startswith("+"):
                is_credit = True

        # Bug 2 fix: clean up description artifacts
        desc = re.sub(r"^\d{1,3}\s+", "", desc)             # strip leading row-index digit(s) ("1 ")
        desc = re.sub(r"^\d{1,2}:\d{2}\s*", "", desc)       # strip leading timestamp
        desc = re.sub(r"\s*HTTPS?://\S*", "", desc, flags=re.IGNORECASE)  # strip URL noise
        desc = re.sub(r"\s+R\s*$", "", desc)                 # strip trailing " R" artifact
        desc = desc.strip(" |-")

        transactions.append({
            "date": date_normalized,
            "description": desc,
            "amount": amount_val,
            "type": "credit" if is_credit else "debit",
        })
        debug_log.append({"row": row_str[:80], "status": "OK"})

    st.session_state["qwen_debug"] = debug_log
    st.session_state["qwen_raw_output"] = transactions
    return transactions



def validate_and_store_transactions(
    parsed_output: list[dict], ocr_row_confidences: list[float],
    original_rows: list[str] | None = None,
) -> pd.DataFrame:
    """Validate regex-parsed transactions and store as DataFrame.

    parse_rows_fast() already handles date normalization, amount cleaning,
    and credit detection via '+' prefix. This function validates fields
    and assigns OCR confidence scores.

    Returns:
        DataFrame with columns: date, description, amount, type, confidence.
        Also stored in st.session_state['df_statements'].
    """
    validated = []
    validation_debug = []

    for idx, txn in enumerate(parsed_output):
        issues = []

        # Validate date
        date_val = txn.get("date", "")
        date_ok = False
        if date_val:
            try:
                pd.to_datetime(date_val, format="%Y-%m-%d")
                date_ok = True
            except (ValueError, TypeError):
                issues.append(f"bad date: {date_val!r}")

        # Validate amount
        amount_valid = True
        amt_val = txn.get("amount", 0)
        try:
            amt = float(amt_val)
            if amt <= 0:
                amount_valid = False
                issues.append(f"amount <= 0: {amt_val!r}")
        except (ValueError, TypeError):
            amount_valid = False
            issues.append(f"bad amount: {amt_val!r}")

        # OCR confidence
        ocr_conf = ocr_row_confidences[idx] if idx < len(ocr_row_confidences) else 0.5

        # Score: penalize missing date or bad amount
        score = ocr_conf
        if not date_ok:
            score *= 0.6
        if not amount_valid:
            score *= 0.3

        validation_debug.append({
            "idx": idx,
            "date": date_val,
            "desc": txn.get("description", "")[:40],
            "amount": amt_val,
            "type": txn.get("type", ""),
            "ocr_conf": round(ocr_conf, 2),
            "final_score": round(score, 2),
            "issues": "; ".join(issues) if issues else "OK",
            "kept": score >= 0.15,
        })

        if score < 0.15:
            continue

        validated.append({
            "date": date_val,
            "description": txn.get("description", ""),
            "amount": float(amt_val) if amount_valid else 0.0,
            "type": txn.get("type", "debit"),
            "confidence": round(score, 4),
        })

    st.session_state["validation_debug"] = validation_debug

    df = pd.DataFrame(
        validated,
        columns=["date", "description", "amount", "type", "confidence"],
    )

    st.session_state["df_statements"] = df
    return df


# ---------------------------------------------------------------------------
# Online Pipeline helpers — HuggingFace Inference API
#   - Receipts:   HF Qwen2.5-VL (vision) reads receipt images directly
#   - Statements: DocTR (local) → build_rows → HF Qwen2.5 (text) parses rows
#   No financial document images leave the machine — only receipt images
#   and extracted row-text strings are sent to HF.
# ---------------------------------------------------------------------------

def _hf_chat(messages: list[dict], max_tokens: int = 2048,
             model: str | None = None) -> str | None:
    """Generic HF Inference API chat-completions call. Returns response text."""
    hf_key = os.environ.get("HF_API_KEY", "")
    if not hf_key:
        return None
    if model is None:
        model = os.environ.get("HF_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    try:
        resp = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {hf_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=180,
        )
        if resp.status_code != 200:
            print(f"HF API error {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
            return None
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"HF API error: {e}", file=sys.stderr)
        return None


def _hf_parse_json(raw_text: str | None) -> list | dict | None:
    """Extract JSON array or object from HF response (handles markdown fences)."""
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    arr_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if arr_match:
        cleaned = arr_match.group(0)
    else:
        obj_match = re.search(r'\{[^{}]*\}', cleaned)
        if obj_match:
            cleaned = obj_match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ── HF Receipt extraction (VLM — sends the receipt IMAGE) ────────────────

def _hf_extract_receipt(image_path: str) -> dict:
    """Extract vendor/amount/date from a receipt image via HF Qwen2.5-VL."""
    import base64
    fallback = {
        "receipt_file": os.path.basename(image_path),
        "vendor": None, "amount": None, "date": None,
        "raw_text": "", "confidence": 0.0, "status": "failed",
    }

    # Encode image as base64 JPEG (resize to max 1024px)
    try:
        img = _load_image_fixed(Path(image_path))
        w, h = img.size
        max_side = 1024
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        fallback["raw_text"] = f"Image encode error: {e}"
        return fallback

    prompt = (
        "Look at this receipt image carefully. Extract these 3 fields.\n"
        'Return ONLY valid JSON: {"vendor": "...", "amount": number, "date": "YYYY-MM-DD"}\n\n'
        "1. vendor: The store/restaurant/company name printed at the top.\n"
        "2. amount: The FINAL TOTAL amount paid (after discounts/taxes). "
        "Look for 'Grand Total', 'Net Amount', 'Total', 'Amount Paid', 'You Pay'. "
        "Do NOT pick subtotal, tax, CGST, SGST, discount, or MRP.\n"
        "3. date: The billing/transaction date. Indian format DD/MM/YYYY → convert to YYYY-MM-DD.\n\n"
        "Use null for any field you cannot read. No explanation, JSON only."
    )

    raw = _hf_chat([{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ],
    }], max_tokens=200)

    parsed = _hf_parse_json(raw)
    if not parsed or not isinstance(parsed, dict):
        fallback["raw_text"] = f"HF VLM returned unparseable response: {(raw or '')[:200]}"
        return fallback

    result = {
        "receipt_file": os.path.basename(image_path),
        "vendor": parsed.get("vendor"),
        "amount": None,
        "date": parsed.get("date"),
        "raw_text": f"[HF VLM extraction]\n{raw}",
        "confidence": 0.90,
        "status": "success",
    }
    try:
        result["amount"] = float(parsed["amount"])
    except (TypeError, ValueError, KeyError):
        pass

    if not result["vendor"] and not result["amount"]:
        result["status"] = "failed"
        result["confidence"] = 0.0

    return result


def hf_extract_receipts_batch(image_paths: list[str],
                              progress_callback=None) -> list[dict]:
    """Process multiple receipts via HF Qwen2.5-VL (sequential)."""
    results = []
    for i, path in enumerate(image_paths):
        data = _hf_extract_receipt(path)
        results.append(data)
        if progress_callback:
            progress_callback(i + 1, len(image_paths))
    return results


# ── HF Statement row parsing (TEXT only — no images sent) ─────────────────

def hf_call_qwen_text(rows: list[str]) -> list[dict]:
    """Send OCR row strings to HF Qwen2.5 text model for structured parsing.

    Same logic as call_qwen() but uses HF Inference API instead of local Ollama.
    Only row TEXT is sent — no images, no financial documents.
    """
    hf_text_model = os.environ.get("HF_TEXT_MODEL",
                                   os.environ.get("HF_VLM_MODEL",
                                                  "Qwen/Qwen2.5-VL-7B-Instruct"))

    all_transactions = []
    debug_log = []
    chunk_size = 20

    st.session_state["qwen_input_rows"] = rows

    for chunk_start in range(0, len(rows), chunk_size):
        chunk = rows[chunk_start:chunk_start + chunk_size]

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
            "- amount (float, always positive)\n\n"
            "Rules:\n"
            "- Dates must be normalized to YYYY-MM-DD (input is DD/MM/YYYY Indian format)\n"
            "- Remove currency symbols and +/- signs from amounts\n"
            "- Ignore invalid/header rows\n"
            "- Each row has token hints after // showing which zone each part belongs to\n\n"
            "Examples:\n"
            'Input: "12/02/2026 | Amazon Pay | 1,200.00  // hints: [DATE_ZONE: 12/02/2026] '
            '[DESC_ZONE: Amazon Pay] [AMOUNT_ZONE: 1,200.00]"\n'
            'Output: {"date":"2026-02-12","description":"Amazon Pay","amount":1200.00}\n\n'
            'Input: "13/02/2026 | Swiggy | +450.00  // hints: [DATE_ZONE: 13/02/2026] '
            '[DESC_ZONE: Swiggy] [AMOUNT_ZONE: +450.00]"\n'
            'Output: {"date":"2026-02-13","description":"Swiggy","amount":450.00}\n\n'
            "Return ONLY a valid JSON array. No explanation.\n\n"
            f"Now process these rows:\n{rows_text}"
        )

        raw = _hf_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=2000,
            model=hf_text_model,
        )

        parsed = _hf_parse_json(raw)

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
                "error": f"Unparseable: {(raw or '')[:300]}",
                "raw_response": (raw or "")[:300],
            })

    st.session_state["qwen_debug"] = debug_log
    st.session_state["qwen_raw_output"] = all_transactions
    return all_transactions


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

    # Process buttons — stacked vertically: Process, Process Online, Clear Cache
    process_clicked = st.button("Process", type="primary", use_container_width=True)
    process_online_clicked = st.button("Process Online", type="secondary", use_container_width=True)
    if st.button("Clear Cache", use_container_width=True):
        # Wipe all cached state so next Process runs fresh
        for k in _PERSIST_DF_KEYS + _PERSIST_SCALAR_KEYS:
            st.session_state.pop(k, None)
        for k in ["qwen_input_rows", "qwen_debug", "qwen_raw_output",
                   "debug_qwen_credits", "validation_debug",
                   "statement_pages", "debug_row_layouts",
                   "debug_row_words", "debug_credit_rows"]:
            st.session_state.pop(k, None)
        _SESSION_CACHE_FILE.unlink(missing_ok=True)
        # Clear statement JSON cache too
        for f in CACHE_DIR_STATEMENTS.glob("*.json"):
            f.unlink(missing_ok=True)
        st.rerun()

# ---- Auto-invalidate stale results when uploaded files change ----
# Track what was uploaded so we don't show stale results from a previous run.
_current_receipt_names = sorted(uf.name for uf in uploaded_receipts) if uploaded_receipts else []
_current_stmt_name = uploaded_statement.name if uploaded_statement else None
_prev_receipt_names = st.session_state.get("_prev_receipt_names", [])
_prev_stmt_name = st.session_state.get("_prev_stmt_name")

if _current_receipt_names != _prev_receipt_names:
    # Receipts changed — clear receipt + match results (statement can stay)
    for k in ["df_receipts", "df_matches", "debug_receipt_ocr"]:
        st.session_state.pop(k, None)
    st.session_state["_prev_receipt_names"] = _current_receipt_names

if _current_stmt_name and _current_stmt_name != _prev_stmt_name:
    # Statement changed — clear statement + match results (receipts can stay)
    for k in ["df_statements", "df_matches", "_cached_statement_name",
              "statement_pages", "debug_row_layouts", "debug_row_words",
              "debug_credit_rows", "qwen_input_rows", "qwen_debug",
              "qwen_raw_output", "debug_qwen_credits", "validation_debug"]:
        st.session_state.pop(k, None)
    st.session_state["_prev_stmt_name"] = _current_stmt_name

# ---- Save uploaded files to disk & process ONLY when Process is clicked ----
receipt_files = []
statement_files = []

if process_clicked or process_online_clicked:
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

        # Show quick status in sidebar
        n_ok = sum(1 for r in receipt_records if r.get("status") == "success")
        n_fail = len(receipt_records) - n_ok
        if n_fail:
            st.sidebar.warning(f"Receipts: {n_ok} success, {n_fail} failed")
            for r in receipt_records:
                if r.get("status") != "success":
                    st.sidebar.caption(f"  {r.get('receipt_file')}: {r.get('raw_text', '')[:200]}")

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
            # Step 2c: Parse rows — columnar parser is primary (uses pipe structure directly)
            # LLM is secondary (only if columnar yields too few results)
            # Regex is last resort
            st.sidebar.info(f"Step 2c — Parsing {len(rows)} rows…")
            try:
                qwen_output = parse_rows_columnar(rows)
                n_col = len(qwen_output)
                st.sidebar.caption(f"  → Columnar parser: {n_col} rows")
                # If columnar got very few results, columns may be badly formed — try LLM
                if n_col < max(1, len(rows) // 3):
                    st.sidebar.warning(f"Columnar got only {n_col}/{len(rows)} — trying LLM…")
                    llm_output = parse_rows_with_llm(rows)
                    if len(llm_output) > n_col:
                        qwen_output = llm_output
                        st.sidebar.caption(f"  → LLM improved to {len(llm_output)} rows")
            except Exception as e:
                st.sidebar.warning(f"Columnar failed ({e}), falling back to regex…")
                qwen_output = parse_rows_fast(rows)

            # DEBUG: Show credit/debit breakdown
            qwen_credits = [
                (i, t.get("description", "")[:30], t.get("amount"), t.get("type"))
                for i, t in enumerate(qwen_output) if t.get("type", "").lower() == "credit"
            ]
            st.session_state["debug_qwen_credits"] = qwen_credits
            st.sidebar.caption(f"  → Parsed {len(qwen_output)} txns ({len(qwen_credits)} credits)")

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
# ONLINE PIPELINE — HF Qwen2.5-VL for receipts, local DocTR + HF text for
# statements.  No financial document images leave the machine.
# ---------------------------------------------------------------------------
if process_online_clicked:
    hf_key_check = os.environ.get("HF_API_KEY", "")
    if not hf_key_check:
        st.sidebar.error("HF_API_KEY not set in .env — cannot run online pipeline.")
    else:
        # ── STEP 1: Process Receipts via HF Qwen2.5-VL ──
        if receipt_files:
            progress = st.sidebar.progress(0, text="[Online] Step 1/2 — Receipts via HF VLM…")

            def _hf_receipt_progress(done, total):
                progress.progress(done / total, text=f"[Online] Receipt {done}/{total} (HF VLM)")

            receipt_records = hf_extract_receipts_batch(
                [str(f) for f in receipt_files],
                progress_callback=_hf_receipt_progress,
            )

            n_ok = sum(1 for r in receipt_records if r.get("status") == "success")
            n_fail = len(receipt_records) - n_ok
            if n_fail:
                st.sidebar.warning(f"[Online] Receipts: {n_ok} success, {n_fail} failed")

            receipt_debug = st.session_state.setdefault("debug_receipt_ocr", {})
            for data in receipt_records:
                receipt_debug[data.get("receipt_file", "")] = {
                    "raw_text": data.get("raw_text", ""),
                    "vendor": data.get("vendor"),
                    "amount": data.get("amount"),
                    "date": data.get("date"),
                    "confidence": data.get("confidence", 0.0),
                    "status": data.get("status", "failed"),
                }
            gc.collect()
            progress.empty()

            df_new = pd.DataFrame(receipt_records)
            if "df_receipts" in st.session_state:
                existing = st.session_state["df_receipts"]
                df_new = pd.concat([existing, df_new]).drop_duplicates(
                    subset="receipt_file", keep="last"
                ).reset_index(drop=True)
            st.session_state["df_receipts"] = df_new
            st.sidebar.success(f"[Online] Step 1 done — {len(receipt_records)} receipt(s) via HF VLM")
            del receipt_records
            gc.collect()
        elif st.session_state.get("df_receipts") is not None:
            st.sidebar.info("[Online] No new receipts. Using previously processed data.")
        else:
            st.sidebar.warning("[Online] No receipt images found.")

        # ── STEP 2: Statements — local DocTR OCR → build_rows → HF text model ──
        _cached_stmt_name_ol = st.session_state.get("_cached_statement_name")
        _current_stmt_name_ol = statement_files[0].name if statement_files else None
        _stmt_cached_ol = (
            _current_stmt_name_ol
            and _current_stmt_name_ol == _cached_stmt_name_ol
            and st.session_state.get("df_statements") is not None
        )

        if not _stmt_cached_ol and _current_stmt_name_ol:
            _cached_df_ol = _load_stmt_cache(_current_stmt_name_ol)
            if _cached_df_ol is not None:
                st.session_state["df_statements"] = _cached_df_ol
                st.session_state["_cached_statement_name"] = _current_stmt_name_ol
                _stmt_cached_ol = True
                st.sidebar.success(f"[Online] Loaded '{_current_stmt_name_ol}' from cache")

        if statement_files and not _stmt_cached_ol:
            # Step 2a: Local DocTR OCR (same subprocess as offline — no data leaves machine)
            progress = st.sidebar.progress(0, text="[Online] Step 2a — Local OCR (DocTR)…")
            all_statement_pages = []
            for idx, sfile in enumerate(statement_files):
                try:
                    pages = process_statement_pdf(str(sfile))
                    all_statement_pages.extend(pages)
                except Exception as e:
                    all_statement_pages.append({
                        "page_number": -1, "raw_ocr_words": [],
                        "status": f"failed: {e}",
                    })
                gc.collect()
                progress.progress((idx + 1) / len(statement_files),
                                  text=f"[Online] OCR page {idx + 1}/{len(statement_files)}")
            progress.empty()
            st.session_state["statement_pages"] = all_statement_pages

            for pg in all_statement_pages:
                pg_num = pg.get("page_number", "?")
                n_words = len(pg.get("raw_ocr_words", []))
                st.sidebar.caption(f"  Page {pg_num}: {pg.get('status', '?')} — {n_words} words")

            # Step 2b: Build rows locally
            st.sidebar.info("[Online] Step 2b — Reconstructing rows…")
            rows, ocr_confidences = build_rows(all_statement_pages)
            del all_statement_pages
            gc.collect()
            st.sidebar.caption(f"  → {len(rows)} row(s) reconstructed")

            if rows:
                # Step 2c: Send row TEXT to HF Qwen2.5 text model (no images)
                st.sidebar.info(f"[Online] Step 2c — Sending {len(rows)} rows to HF Qwen2.5…")
                qwen_output = hf_call_qwen_text(rows)
                st.sidebar.caption(
                    f"  → HF returned {len(qwen_output)} transaction(s)"
                )

                # Step 2d: Validate
                st.sidebar.info("[Online] Step 2d — Validating transactions…")
                df_statements = validate_and_store_transactions(
                    qwen_output, ocr_confidences, rows
                )

                st.session_state["_cached_statement_name"] = _current_stmt_name_ol
                _save_stmt_cache(_current_stmt_name_ol, df_statements)

                st.sidebar.success(
                    f"[Online] Step 2 done — {len(df_statements)} transaction(s) via HF text model"
                )
                del qwen_output, rows, ocr_confidences
                gc.collect()
            else:
                st.sidebar.warning("[Online] No rows reconstructed from statements.")
        elif _stmt_cached_ol:
            st.sidebar.success(
                f"[Online] Statement '{_current_stmt_name_ol}' already cached — "
                f"{len(st.session_state['df_statements'])} txn(s)"
            )
        elif st.session_state.get("df_statements") is not None:
            st.sidebar.info("[Online] No new statements. Using previously processed data.")
        else:
            st.sidebar.info("[Online] No statements uploaded.")

        # ── STEP 3: Match (same as offline) ──
        if "df_receipts" in st.session_state and "df_statements" in st.session_state:
            st.sidebar.info("[Online] Step 3 — Matching receipts to transactions…")
            df_matches = match_transactions(
                st.session_state["df_receipts"],
                st.session_state["df_statements"],
            )
            st.session_state["df_matches"] = df_matches
            n_auto = len(df_matches[df_matches["status"] == "auto_approved"])
            n_review = len(df_matches[df_matches["status"] == "review"])
            n_unmatched = len(df_matches[df_matches["match_score"] == 0])

            st.sidebar.success(
                f"[Online] Step 3 done — {n_auto} auto, {n_review} review, {n_unmatched} unmatched"
            )

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

tab_high, tab_review, tab_unmatched, tab_credits, tab_compare = st.tabs(
    [
        "Auto-Approved",
        "Needs Review",
        "Unmatched",
        "Credits",
        "Receipt vs Transaction",
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

    # ── Statement Debug: OCR Raw Rows → Parsed Result side by side ──
    st.markdown("---")
    st.subheader("Statement Debug")

    _ocr_rows = st.session_state.get("qwen_input_rows", [])
    _parsed_txns = st.session_state.get("qwen_raw_output", [])

    if _ocr_rows or _parsed_txns:
        # Show OCR raw rows (what build_rows produced)
        with st.expander(f"OCR Raw Rows ({len(_ocr_rows)} rows)", expanded=False):
            if _ocr_rows:
                for i, r in enumerate(_ocr_rows):
                    st.text(f"{i + 1:3d}. {r}")
            else:
                st.info("No OCR rows.")

        # Show parsed result (what regex extracted)
        st.markdown(f"**Parsed Transactions ({len(_parsed_txns)})**")
        if _parsed_txns:
            st.dataframe(pd.DataFrame(_parsed_txns), hide_index=True, use_container_width=True)
        else:
            st.info("No transactions parsed.")

        # Side-by-side comparison: raw row vs parsed
        with st.expander("Row-by-Row Comparison (OCR vs Parsed)", expanded=True):
            parse_idx = 0
            for i, raw_row in enumerate(_ocr_rows):
                if parse_idx < len(_parsed_txns):
                    p = _parsed_txns[parse_idx]
                    st.markdown(
                        f"**Row {i + 1}**  \n"
                        f"OCR: `{raw_row[:100]}`  \n"
                        f"→ Date: `{p.get('date', '')}` | "
                        f"Desc: `{p.get('description', '')[:50]}` | "
                        f"Amt: `{p.get('amount', '')}` | "
                        f"Type: `{p.get('type', '')}`"
                    )
                    parse_idx += 1
                else:
                    st.markdown(
                        f"**Row {i + 1}**  \n"
                        f"OCR: `{raw_row[:100]}`  \n"
                        f"→ *SKIPPED by parser*"
                    )
                st.markdown("---")
    elif _df_statements is not None and not _df_statements.empty:
        st.markdown("**Final Statement Transactions**")
        st.dataframe(_df_statements, hide_index=True, use_container_width=True)
    else:
        st.info("Process a statement to see debug output here.")

    # ── Receipt OCR Debug ──
    st.markdown("---")
    st.subheader("Receipt OCR Debug")
    _receipt_debug_all = st.session_state.get("debug_receipt_ocr", {})
    if _receipt_debug_all:
        # Summary table
        _debug_rows = []
        for rname, rdbg in _receipt_debug_all.items():
            _debug_rows.append({
                "Receipt": rname,
                "Status": rdbg.get("status", "failed"),
                "Vendor": rdbg.get("vendor") or "—",
                "Amount": rdbg.get("amount") or "—",
                "Date": rdbg.get("date") or "—",
                "Confidence": f"{rdbg.get('confidence', 0.0):.0%}",
            })
        st.dataframe(pd.DataFrame(_debug_rows), width="stretch", hide_index=True)

        # Per-receipt detail
        for receipt_name, dbg in _receipt_debug_all.items():
            img_path = UPLOAD_DIR_RECEIPTS / receipt_name
            with st.expander(f"{receipt_name} — {dbg.get('status', 'failed')}"):
                col_img, col_result = st.columns([1, 2])
                with col_img:
                    if img_path.exists():
                        st.image(_load_image_fixed(img_path), caption=receipt_name,
                                 use_container_width=True)
                with col_result:
                    st.markdown(
                        f"**Vendor:** {dbg.get('vendor') or '—'}  \n"
                        f"**Amount:** {dbg.get('amount') or '—'}  \n"
                        f"**Date:** {dbg.get('date') or '—'}  \n"
                        f"**Confidence:** {dbg.get('confidence', 0.0):.0%}"
                    )
                    raw_text = dbg.get("raw_text", "")
                    if raw_text:
                        st.text_area("Raw Output", value=raw_text, height=150,
                                     key=f"ocr_debug_{receipt_name}", disabled=True)
    else:
        st.info("Process receipts to see OCR debug output.")
