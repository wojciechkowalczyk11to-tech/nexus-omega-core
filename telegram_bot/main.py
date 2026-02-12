"""
Telegram bot main entry point.
Will be fully implemented in Phase 5.
"""

import asyncio
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Main bot entry point."""
    logger.info("Telegram bot starting (stub - Phase 5 implementation pending)...")
    # Keep container alive
    while True:
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
