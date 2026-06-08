-- ═══════════════════════════════════════════════════════════════════════════
-- ModelMesh — PostgreSQL Initialization Script
-- Runs once on first container start via docker-entrypoint-initdb.d/
-- ═══════════════════════════════════════════════════════════════════════════

-- Enable UUID extension (used as primary keys)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pg_stat_statements for query performance monitoring
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ── Performance tuning for time-series prediction logs ─────────────────────
-- These settings are appropriate for a 4-8 core, 8GB RAM machine.
-- Adjust based on your actual hardware.

ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';

-- ── MLflow Database (separate schema to avoid table conflicts) ──────────────
-- MLflow stores its data in the same DB using its own tables.
-- We isolate our tables in the 'modelmesh' schema.
CREATE SCHEMA IF NOT EXISTS modelmesh;

-- ── Grant permissions ──────────────────────────────────────────────────────
GRANT ALL PRIVILEGES ON DATABASE modelmesh_db TO modelmesh;
GRANT ALL PRIVILEGES ON SCHEMA modelmesh TO modelmesh;
GRANT ALL PRIVILEGES ON SCHEMA public TO modelmesh;

-- ── Create read-only reporting user ───────────────────────────────────────
-- Used by Grafana and reporting tools — no write access
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'modelmesh_readonly') THEN
        CREATE ROLE modelmesh_readonly WITH LOGIN PASSWORD 'readonly_secret';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE modelmesh_db TO modelmesh_readonly;
GRANT USAGE ON SCHEMA public TO modelmesh_readonly;

-- Read access will be granted after tables are created by SQLAlchemy init_db()
-- Run this after first startup:
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO modelmesh_readonly;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO modelmesh_readonly;

-- ── Confirmation ──────────────────────────────────────────────────────────
DO $$
BEGIN
    RAISE NOTICE 'ModelMesh database initialized successfully';
END
$$;
