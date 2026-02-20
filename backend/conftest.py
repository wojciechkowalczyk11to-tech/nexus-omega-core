"""
Root-level conftest for backend tests.

Sets required environment variables BEFORE any app modules are imported,
because app/core/config.py executes `settings = Settings()` at module level.
"""

import os

# --------------------------------------------------------------------------
# Required settings (no defaults in Settings model)
# --------------------------------------------------------------------------
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1234567890:AAFakeTokenForTestingOnly_DoNotUse")
os.environ.setdefault("DEMO_UNLOCK_CODE", "TEST_UNLOCK_CODE")
os.environ.setdefault("BOOTSTRAP_ADMIN_CODE", "TEST_ADMIN_CODE")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_min_32_chars_long_enough")
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost:5432/nexus_test"
)
os.environ.setdefault("POSTGRES_PASSWORD", "password")

# --------------------------------------------------------------------------
# Optional settings (have defaults, but set to avoid warnings)
# --------------------------------------------------------------------------
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("LOG_JSON", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
