"""
Health check endpoint for monitoring service status.
"""

from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_db, get_redis
from backend.app.core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health")
async def health_check(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> dict[str, str]:
    """
    Health check endpoint.

    Checks:
    - Database connectivity
    - Redis connectivity
    - Service status

    Returns:
        Health status response
    """
    health_status = {
        "status": "healthy",
        "database": "unknown",
        "redis": "unknown",
    }

    # Check database
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        health_status["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["database"] = "unhealthy"
        health_status["status"] = "unhealthy"

    # Check Redis
    try:
        await redis.ping()
        health_status["redis"] = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["redis"] = "unhealthy"
        health_status["status"] = "unhealthy"

    return health_status
