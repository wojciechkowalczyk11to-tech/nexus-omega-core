"""
Celery background tasks.
"""

import asyncio
import os
import shutil
import tempfile
from datetime import UTC
from typing import Any

from app.core.logging_config import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="downgrade_expired_subscriptions")
def downgrade_expired_subscriptions() -> int:
    """Batch downgrade all users with expired subscriptions."""
    from datetime import datetime

    from sqlalchemy import update

    from app.db.models.user import User
    from app.db.session import async_session_maker

    async def _run() -> int:
        async with async_session_maker() as db:
            result = await db.execute(
                update(User)
                .where(
                    User.role == "FULL_ACCESS",
                    User.subscription_expires_at < datetime.now(UTC),
                )
                .values(role="DEMO", subscription_tier=None)
                .returning(User.telegram_id)
            )
            downgraded = result.scalars().all()
            await db.commit()
            if downgraded:
                logger.info(
                    "Batch downgraded %d expired subscriptions: %s",
                    len(downgraded),
                    downgraded,
                )
            return len(downgraded)

    return asyncio.run(_run())


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

        logger.info("Cleaned up %s old sessions", deleted_count)

        return {
            "status": "success",
            "deleted_count": deleted_count,
        }

    except Exception as e:
        logger.error("cleanup_old_sessions error: %s", e, exc_info=True)
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
    logger.info("Generating usage report for user %s, period=%s days", user_id, period_days)

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

        logger.info("Generated usage report for user %s: %s", user_id, report)

        return {
            "status": "success",
            "report": report,
        }

    except Exception as e:
        logger.error("generate_usage_report error: %s", e, exc_info=True)
        return {
            "status": "error",
            "error": str(e),
        }


@celery_app.task(name="tasks.sync_github_repo")
def sync_github_repo(user_id: int, repo_url: str) -> dict[str, Any]:
    """
    Sync GitHub repository for Devin mode.

    Clones the repository, indexes code files into the RAG system
    with pgvector embeddings for semantic search.

    Args:
        user_id: User's Telegram ID
        repo_url: GitHub repository URL

    Returns:
        Task result dict with status, repo_url, and files_indexed count
    """
    logger.info("Syncing GitHub repo for user %s: %s", user_id, repo_url)

    temp_repo_dir = ""

    try:
        from app.db.session import AsyncSessionLocal
        from app.tools.github_devin_tool import GitHubDevinTool

        temp_repo_dir = tempfile.mkdtemp(prefix=f"github_sync_{user_id}_")

        async def _sync() -> int:
            async with AsyncSessionLocal() as session:
                github_tool = GitHubDevinTool(user_id=user_id, db=session)
                github_tool.sandbox.repos_dir = temp_repo_dir

                clone_result = await github_tool.clone_repository(repo_url)
                index_result = await github_tool.index_repository(clone_result["repo_name"])

                await session.commit()
                return index_result.get("files_indexed", 0)

        files_indexed = asyncio.run(_sync())

        return {
            "status": "success",
            "repo_url": repo_url,
            "files_indexed": files_indexed,
        }

    except Exception as e:
        logger.error("sync_github_repo error: %s", e, exc_info=True)
        return {
            "status": "error",
            "error": str(e),
        }
    finally:
        if temp_repo_dir and os.path.exists(temp_repo_dir):
            shutil.rmtree(temp_repo_dir, ignore_errors=True)
