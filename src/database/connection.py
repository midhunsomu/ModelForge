"""
ModelMesh — Async Database Connection Management
-------------------------------------------------
Uses SQLAlchemy async engine with asyncpg driver.
Connection pool is sized to handle burst inference traffic.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────────────────
# pool_pre_ping=True: validates connections before use, handles stale sockets
# pool_size: baseline connections kept alive
# max_overflow: extra connections allowed under burst load

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database.url,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,          # recycle connections every hour
            echo=settings.debug,
        )
        logger.info(
            "database_engine_created",
            host=settings.database.host,
            pool_size=settings.database.pool_size,
        )
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,  # prevent lazy-load errors after commit
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.
    Always commits on success, rolls back on any exception.
    Usage:
        async with get_db_session() as session:
            result = await session.execute(...)
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error("db_session_rollback", error=str(exc))
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency injection version."""
    async with get_db_session() as session:
        yield session


async def init_db() -> None:
    """Create all tables on startup (idempotent)."""
    from src.database.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_schema_initialized")


async def check_db_health() -> dict:
    """Health check: verify DB connection is alive."""
    try:
        async with get_db_session() as session:
            result = await session.execute(text("SELECT 1"))
            result.fetchone()
        return {"status": "healthy", "latency_ms": 0}
    except Exception as exc:
        logger.error("db_health_check_failed", error=str(exc))
        return {"status": "unhealthy", "error": str(exc)}


async def close_db() -> None:
    """Gracefully dispose the engine on shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("database_engine_closed")
