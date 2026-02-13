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
def sync_github_repo(repo_url: str, user_id: int) -> dict[str, Any]:
    """
    Sync GitHub repository for Devin mode.

    Clones the repository, indexes code files into the RAG system
    with pgvector embeddings for semantic search.

    Args:
        repo_url: GitHub repository URL
        user_id: User's Telegram ID

    Returns:
        Task result dict with status, repo_url, and files_indexed count
    """
    import shutil
    import tempfile

    logger.info(f"Syncing GitHub repo for user {user_id}: {repo_url}")

    clone_dir = tempfile.mkdtemp(prefix="github_sync_")

    try:
        from app.db.session import async_session_maker
        from app.tools.github_devin_tool import GitHubDevinTool

        async def _sync() -> dict[str, Any]:
            async with async_session_maker() as db:
                tool = GitHubDevinTool(user_id=user_id, db=db)
                clone_result = await tool.clone_repository(repo_url)
                repo_name = clone_result["repo_name"]
                index_result = await tool.index_repository(repo_name)
                await db.commit()
                return {
                    "repo_name": repo_name,
                    "files_indexed": index_result["files_indexed"],
                }

        result = asyncio.run(_sync())

        logger.info(
            f"GitHub repo synced for user {user_id}: "
            f"{result['files_indexed']} files indexed"
        )

        return {
            "status": "success",
            "repo_url": repo_url,
            "files_indexed": result["files_indexed"],
        }

    except Exception as e:
        logger.error(f"sync_github_repo error: {e}", exc_info=True)
        return {
            "status": "error",
            "repo_url": repo_url,
            "error": str(e),
        }
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)
