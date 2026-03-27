# CLAUDE.md — Project Context for AI Assistant
Codex will review your output once you are done.
## Project Overview
AI-powered expense reconciliation system that automates matching receipt images to credit card/bank statement transactions.

**Pipeline:** Receipt Upload → Receipt OCR → Statement OCR → Matching → Human Review → Export

## Tech Stack
- **UI:** Streamlit (`app.py`) — exploring Flask migration for better UI
- **Receipt OCR:** PaddleOCR v3.4 (PP-OCRv4, `ocr_receipt.py`)
- **Statement OCR:** python-doctr + Qwen LLM parsing (`ocr_statement.py`)
- **LLM:** Gemini API (amount extraction, receipt parsing)
- **Matching:** rapidfuzz (vendor name matching), date/amount heuristics
- **Storage:** PostgreSQL, disk cache for session persistence
- **Other:** pdf2image, opencv-python, Pillow, pandas, python-dateutil

## Project Structure
```
D:\Office Work\Project\
├── app.py              # Main Streamlit application (~152KB, large file)
├── ocr_receipt.py      # Receipt OCR pipeline (PaddleOCR + Gemini)
├── ocr_statement.py    # Statement OCR pipeline (doctr + Qwen LLM)
├── .env                # API keys (Gemini, etc.) — DO NOT commit
├── requirements.txt    # Python dependencies
├── setup.bat           # Windows setup script
├── cache/              # Session persistence cache
├── db/                 # Database related files
├── uploads/            # Uploaded receipt/statement files
├── poppler_install/    # PDF rendering dependency
├── venv/               # Python virtual environment
└── .claude/            # Claude memory files
```

## Key Design Decisions
1. **Amount extraction:** Must find FINAL PAYABLE amount (Net Rs), not subtotal. Gemini LLM is authoritative source.
2. **PaddleOCR config:** Use PP-OCRv4 (not v5), `enable_mkldnn=False`, `predict()` not `ocr()`. See memory for exact constructor params.
3. **Session persistence:** App state cached to `cache/session/last_session.json` to survive browser tab switches.
4. **Statement OCR:** Uses dynamic Y-clustering + vertical welding for row building, raw OCR rows passed to Qwen LLM for parsing.
5. **Credit detection:** Keyword-based (AUTOPAY, REFUND, etc.) + green text color detection for credit amounts.

## Git Info
- **Branch:** `laptop` (working branch)
- **Main branch:** `main`
- **Remote:** GitHub (krishnakotecha19)

## Current Status (as of 2026-03-26)
- Receipt OCR + Statement OCR pipelines working
- Statement matching with OCR functional
- Statement amounts formatted with 2 decimal places
- Credit detection via keyword + color detection implemented
- Considering Flask UI migration for more polished, professional look

## Rules
- NEVER push or commit without explicit user permission
- NEVER modify `.env` or credentials files
- Test locally before any branch changes
- Amount extraction must prioritize "Net Rs" / "Net Total" over "Total RS"
- PaddleOCR: always use validated constructor params (see memory/feedback_paddleocr.md)
