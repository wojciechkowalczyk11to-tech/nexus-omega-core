"""
NexusOmegaCore Telegram Bot - Main entry point.
"""

import logging
import signal
import threading
from types import FrameType

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from telegram_bot.config import settings
from telegram_bot.handlers.chat_handler_streaming import chat_message_streaming
from telegram_bot.handlers.document_handler import document_handler
from telegram_bot.handlers.help_handler import help_command
from telegram_bot.handlers.mode_handler import mode_command
from telegram_bot.handlers.provider_handler import provider_command
from telegram_bot.handlers.start_handler import start_command
from telegram_bot.handlers.subscribe_handler import (
    buy_command,
    precheckout_callback,
    subscribe_command,
    successful_payment_callback,
)
from telegram_bot.handlers.unlock_handler import unlock_command

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, settings.log_level.upper()),
)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Start the Telegram bot.
    """
    logger.info("Starting NexusOmegaCore Telegram Bot...")

    if settings.telegram_dry_run:
        logger.info(
            "Telegram dry-run mode enabled; skipping Telegram network startup "
            "for deterministic CI boot verification."
        )
        stop_event = threading.Event()

        def _handle_signal(signum: int, _frame: FrameType | None) -> None:
            logger.info("Dry-run mode received signal %s, shutting down.", signum)
            stop_event.set()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
        stop_event.wait()
        return

    # Create application
    application = Application.builder().token(settings.telegram_bot_token).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("mode", mode_command))
    application.add_handler(CommandHandler("provider", provider_command))
    application.add_handler(CommandHandler("unlock", unlock_command))
    application.add_handler(CommandHandler("subscribe", subscribe_command))
    application.add_handler(CommandHandler("buy", buy_command))

    # Register payment handlers
    from telegram.ext import PreCheckoutQueryHandler

    application.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_callback))

    # Register message handlers
    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_message_streaming))

    # Start polling with graceful shutdown support for Cloud Run (SIGTERM)
    logger.info("Bot started successfully. Polling for updates...")
    application.run_polling(
        allowed_updates=["message", "callback_query"],
        stop_signals=(signal.SIGINT, signal.SIGTERM),
    )


if __name__ == "__main__":
    main()
