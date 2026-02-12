"""
FastAPI application factory with lifespan management.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.deps import close_redis_pool, get_redis_pool
from backend.app.api.v1.router import api_router
from backend.app.core.config import settings
from backend.app.core.logging_config import get_logger, setup_logging
from backend.app.db.session import close_db, init_db

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for startup and shutdown events.

    Handles:
    - Database connection initialization
    - Redis connection pool creation
    - Graceful shutdown
    """
    logger.info("Starting NexusOmegaCore backend...")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Initialize Redis
    try:
        await get_redis_pool()
        logger.info("Redis connection pool created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        raise

    logger.info("NexusOmegaCore backend started successfully")

    yield

    # Shutdown
    logger.info("Shutting down NexusOmegaCore backend...")

    try:
        await close_redis_pool()
        logger.info("Redis connection pool closed")
    except Exception as e:
        logger.error(f"Error closing Redis pool: {e}")

    try:
        await close_db()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")

    logger.info("NexusOmegaCore backend shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="NexusOmegaCore API",
        description="Telegram AI Aggregator Bot Backend",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(api_router, prefix="/api/v1")

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
