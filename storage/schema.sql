-- Crypto Vol Surface Builder — TimescaleDB schema
-- Runs automatically on first container creation (see docker-compose.yml).
--
-- Design principle: a hard line between INGESTION (raw observations, never
-- modified by analysis) and ANALYSIS (versioned pricer output). Raw quotes
-- are the source of truth; everything else is reproducible from them.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- INGESTION — captures what the market showed. Never modified by analysis.
-- ============================================================

-- Static-ish metadata. One row per instrument, ever.
CREATE TABLE IF NOT EXISTS instruments (
    instrument_name   TEXT PRIMARY KEY,           -- 'BTC-26JUN26-77000-C'
    currency          TEXT NOT NULL,              -- 'BTC'
    strike            DOUBLE PRECISION NOT NULL,
    option_type       CHAR(1) NOT NULL,           -- 'C' / 'P'
    expiry            TIMESTAMPTZ NOT NULL,
    contract_size     DOUBLE PRECISION,
    creation_ts       TIMESTAMPTZ
);

-- The vol data. Hypertable. One row per (instrument, snapshot).
CREATE TABLE IF NOT EXISTS option_quotes (
    time              TIMESTAMPTZ NOT NULL,
    instrument_name   TEXT NOT NULL REFERENCES instruments,
    -- raw, source of truth
    mark_price        DOUBLE PRECISION,           -- in BTC (inverse contract)
    best_bid          DOUBLE PRECISION,
    best_ask          DOUBLE PRECISION,
    open_interest     DOUBLE PRECISION,
    -- Deribit's published values: our validation answer key
    deribit_mark_iv   DOUBLE PRECISION,
    deribit_delta     DOUBLE PRECISION,
    PRIMARY KEY (instrument_name, time)
);
SELECT create_hypertable('option_quotes', 'time', if_not_exists => TRUE);

-- Per-expiry forward. Small, normalized. The ATM reference for moneyness.
CREATE TABLE IF NOT EXISTS forwards (
    time              TIMESTAMPTZ NOT NULL,
    expiry            TIMESTAMPTZ NOT NULL,
    forward_price     DOUBLE PRECISION NOT NULL,
    index_price       DOUBLE PRECISION NOT NULL,  -- forward-collector's own index obs
    PRIMARY KEY (expiry, time)
);
SELECT create_hypertable('forwards', 'time', if_not_exists => TRUE);

-- Perp funding, for building our own forward curve in weeks 3-4.
CREATE TABLE IF NOT EXISTS funding_rates (
    time              TIMESTAMPTZ NOT NULL,
    instrument_name   TEXT NOT NULL,              -- 'BTC-PERPETUAL'
    funding_rate_8h   DOUBLE PRECISION,
    index_price       DOUBLE PRECISION,           -- funding-collector's own index obs
    PRIMARY KEY (instrument_name, time)
);
SELECT create_hypertable('funding_rates', 'time', if_not_exists => TRUE);

-- ============================================================
-- ANALYSIS — your pricer's output. Versioned. Regenerated on demand.
-- ============================================================

CREATE TABLE IF NOT EXISTS computed_iv (
    time              TIMESTAMPTZ NOT NULL,
    instrument_name   TEXT NOT NULL,
    pricer_version    TEXT NOT NULL,              -- 'v0.3.1' — compare versions side by side
    my_iv             DOUBLE PRECISION,
    my_delta          DOUBLE PRECISION,
    my_vega           DOUBLE PRECISION,
    iv_error_vs_mark  DOUBLE PRECISION,           -- my_iv - deribit_mark_iv (the grade)
    PRIMARY KEY (instrument_name, time, pricer_version)
);
SELECT create_hypertable('computed_iv', 'time', if_not_exists => TRUE);

-- Helpful indexes for the queries you'll actually run
CREATE INDEX IF NOT EXISTS idx_quotes_time ON option_quotes (time DESC);
CREATE INDEX IF NOT EXISTS idx_instruments_expiry ON instruments (expiry);
