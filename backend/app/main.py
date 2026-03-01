"""
FastAPI application factory with lifespan management.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import close_redis_pool, get_redis_pool
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging_config import get_logger, setup_logging
from app.db.session import close_db, init_db

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
        logger.error("Failed to initialize database: %s", e)
        raise

    # Initialize Redis
    try:
        await get_redis_pool()
        logger.info("Redis connection pool created successfully")
    except Exception as e:
        logger.error("Failed to initialize Redis: %s", e)
        raise

    logger.info("NexusOmegaCore backend started successfully")

    yield

    # Shutdown
    logger.info("Shutting down NexusOmegaCore backend...")

    try:
        await close_redis_pool()
        logger.info("Redis connection pool closed")
    except Exception as e:
        logger.error("Error closing Redis pool: %s", e)

    try:
        await close_db()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error("Error closing database: %s", e)

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
    origins = settings.cors_allowed_origins
    if not origins and settings.environment == "development":
        origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
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
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
