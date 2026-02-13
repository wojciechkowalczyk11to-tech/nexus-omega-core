"""
Celery background tasks.
"""

import asyncio
from datetime import UTC
from typing import Any

from app.core.logging_config import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="tasks.cleanup_old_sessions")
def cleanup_old_sessions() -> dict[str, Any]:
    """
    Cleanup old inactive sessions (older than 30 days).

    Returns:
        Task result dict
    """
    logger.info("Starting cleanup_old_sessions task")

    try:
        # Import here to avoid circular imports
        from datetime import datetime, timedelta

        from sqlalchemy import delete

        from app.db.models.session import ChatSession
        from app.db.session import async_session_maker

        async def _cleanup() -> int:
            cutoff_date = datetime.now(UTC) - timedelta(days=30)

            async with async_session_maker() as session:
                result = await session.execute(
                    delete(ChatSession).where(ChatSession.updated_at < cutoff_date)
                )
                deleted_count = result.rowcount
                await session.commit()

            return deleted_count

        deleted_count = asyncio.run(_cleanup())

        logger.info(f"Cleaned up {deleted_count} old sessions")

        return {
            "status": "success",
            "deleted_count": deleted_count,
        }

    except Exception as e:
        logger.error(f"cleanup_old_sessions error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
        }


@celery_app.task(name="tasks.generate_usage_report")
def generate_usage_report(user_id: int, period_days: int = 30) -> dict[str, Any]:
    """
    Generate usage report for user.

    Args:
        user_id: User's Telegram ID
        period_days: Report period in days

    Returns:
        Task result dict with report data
    """
    logger.info(f"Generating usage report for user {user_id}, period={period_days} days")

    try:
        from datetime import datetime, timedelta

        from sqlalchemy import func, select

        from app.db.models.ledger import UsageLedger
        from app.db.session import async_session_maker

        async def _generate() -> dict[str, Any]:
            cutoff_date = datetime.now(UTC) - timedelta(days=period_days)

            async with async_session_maker() as session:
                # Get usage stats
                result = await session.execute(
                    select(
                        func.count(UsageLedger.id).label("total_requests"),
                        func.sum(UsageLedger.input_tokens).label("total_input_tokens"),
                        func.sum(UsageLedger.output_tokens).label("total_output_tokens"),
                        func.sum(UsageLedger.cost_usd).label("total_cost_usd"),
                    ).where(
                        UsageLedger.user_id == user_id,
                        UsageLedger.created_at >= cutoff_date,
                    )
                )
                stats = result.one()

            return {
                "user_id": user_id,
                "period_days": period_days,
                "total_requests": stats.total_requests or 0,
                "total_input_tokens": stats.total_input_tokens or 0,
                "total_output_tokens": stats.total_output_tokens or 0,
                "total_cost_usd": float(stats.total_cost_usd or 0),
            }

        report = asyncio.run(_generate())

        logger.info(f"Generated usage report for user {user_id}: {report}")

        return {
            "status": "success",
            "report": report,
        }

    except Exception as e:
        logger.error(f"generate_usage_report error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
        }


@celery_app.task(name="tasks.sync_github_repo")
def sync_github_repo(user_id: int, repo_url: str) -> dict[str, Any]:
    """
    Sync GitHub repository for Devin mode.

    Args:
        user_id: User's Telegram ID
        repo_url: GitHub repository URL

    Returns:
        Task result dict
    """
    logger.info(f"Syncing GitHub repo for user {user_id}: {repo_url}")

    try:
        # Placeholder for GitHub sync logic
        # In production, this would:
        # 1. Clone/pull repository
        # 2. Index code files
        # 3. Store in RAG system
        # 4. Update sync status

        return {
            "status": "success",
            "repo_url": repo_url,
            "files_indexed": 0,
            "message": "GitHub sync not implemented (placeholder)",
        }

    except Exception as e:
        logger.error(f"sync_github_repo error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
        }
