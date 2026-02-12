"""
Chat message handler.
"""

from telegram import Update
from telegram.ext import ContextTypes

from telegram_bot.services.backend_client import BackendClient
from telegram_bot.services.user_cache import UserCache


async def chat_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle chat messages.

    Sends message to backend API and returns AI response.
    """
    user = update.effective_user
    query = update.message.text

    backend = BackendClient()
    cache = UserCache()

    try:
        # Get cached token
        token = await cache.get_user_token(user.id)

        if not token:
            await update.message.reply_text(
                "⚠️ Nie jesteś zalogowany. Użyj /start aby się zarejestrować."
            )
            return

        # Check rate limit
        count = await cache.increment_rate_limit(user.id, window=60)
        if count > 30:
            await update.message.reply_text(
                "⚠️ Przekroczono limit zapytań (30/min). Spróbuj za chwilę."
            )
            return

        # Get user mode
        mode = await cache.get_user_mode(user.id)

        # Send typing indicator
        await update.message.chat.send_action("typing")

        # Send to backend
        response = await backend.chat(
            token=token,
            query=query,
            mode=mode,
        )

        # Check if needs confirmation
        if response.get("needs_confirmation"):
            await update.message.reply_text(
                "⚠️ **Tryb DEEP** wymaga potwierdzenia (wyższy koszt).\n\n"
                "Użyj /deep_confirm aby potwierdzić, lub /mode eco aby zmienić tryb.",
                parse_mode="Markdown",
            )
            # Store query in context for confirmation
            context.user_data["pending_query"] = query
            return

        # Format response
        content = response["content"]
        meta_footer = response.get("meta_footer", "")

        # Truncate if too long (Telegram limit: 4096 chars)
        max_length = 4000
        if len(content) > max_length:
            content = content[:max_length] + "\n\n... (odpowiedź skrócona)"

        full_response = f"{content}\n\n---\n{meta_footer}"

        await update.message.reply_text(full_response, parse_mode="Markdown")

    except Exception as e:
        error_message = str(e)

        # Handle specific errors
        if "403" in error_message:
            await update.message.reply_text(
                "⛔ Brak dostępu. Sprawdź swoją rolę: /help"
            )
        elif "503" in error_message:
            await update.message.reply_text(
                "❌ Wszystkie providery AI są niedostępne. Spróbuj ponownie za chwilę."
            )
        else:
            await update.message.reply_text(
                f"❌ Błąd: {error_message}\n\nSpróbuj ponownie lub użyj /help"
            )

    finally:
        await backend.close()
        await cache.close()
