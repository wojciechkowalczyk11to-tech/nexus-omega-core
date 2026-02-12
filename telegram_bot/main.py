"""
NexusOmegaCore Telegram Bot - Main entry point.
"""

import logging

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from telegram_bot.config import settings
from telegram_bot.handlers.chat_handler import chat_message
from telegram_bot.handlers.document_handler import document_handler
from telegram_bot.handlers.help_handler import help_command
from telegram_bot.handlers.mode_handler import mode_command
from telegram_bot.handlers.start_handler import start_command
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

    # Create application
    application = Application.builder().token(settings.telegram_bot_token).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("mode", mode_command))
    application.add_handler(CommandHandler("unlock", unlock_command))

    # Register message handlers
    application.add_handler(
        MessageHandler(filters.Document.ALL, document_handler)
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, chat_message)
    )

    # Start polling
    logger.info("Bot started successfully. Polling for updates...")
    application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
