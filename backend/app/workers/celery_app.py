"""
Celery application configuration.
Will be fully implemented in Phase 6.
"""

from celery import Celery

from backend.app.core.config import settings

celery_app = Celery(
    "nexus_omega_core",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Task discovery will be added in Phase 6
# celery_app.autodiscover_tasks(['backend.app.workers'])
