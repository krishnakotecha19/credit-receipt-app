-- Expense Reconciliation System - PostgreSQL Schema

CREATE TABLE IF NOT EXISTS receipts (
    id              SERIAL PRIMARY KEY,
    file_name       VARCHAR(255) NOT NULL UNIQUE,
    file_data       BYTEA NOT NULL,
    uploaded_at     TIMESTAMP DEFAULT NOW(),
    ocr_status      VARCHAR(20) DEFAULT 'pending',   -- pending | processed | failed
    extracted_vendor VARCHAR(255),
    extracted_amount DECIMAL(12,2),
    extracted_date  DATE,
    raw_ocr_text    TEXT,
    confidence      FLOAT
);

CREATE TABLE IF NOT EXISTS statements (
    id              SERIAL PRIMARY KEY,
    file_name       VARCHAR(255) NOT NULL UNIQUE,
    file_data       BYTEA NOT NULL,
    uploaded_at     TIMESTAMP DEFAULT NOW(),
    ocr_status      VARCHAR(20) DEFAULT 'pending'    -- pending | processed | failed
);

CREATE TABLE IF NOT EXISTS transactions (
    id              SERIAL PRIMARY KEY,
    statement_id    INTEGER NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
    transaction_date DATE,
    description     TEXT,
    amount          DECIMAL(12,2),
    txn_type        VARCHAR(10),                      -- debit | credit | unknown
    raw_text        TEXT,
    row_index       INTEGER,
    confidence      FLOAT
);

CREATE TABLE IF NOT EXISTS matches (
    id              SERIAL PRIMARY KEY,
    receipt_id      INTEGER NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
    transaction_id  INTEGER NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    confidence_score FLOAT,
    status          VARCHAR(20) DEFAULT 'pending',    -- auto_approved | review | rejected | manual_match
    reviewed_by     VARCHAR(100),
    reviewed_at     TIMESTAMP,
    UNIQUE(receipt_id, transaction_id)
);
