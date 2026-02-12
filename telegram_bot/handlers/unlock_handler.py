"""
/unlock command handler.
"""

from telegram import Update
from telegram.ext import ContextTypes

from telegram_bot.services.backend_client import BackendClient
from telegram_bot.services.user_cache import UserCache


async def unlock_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /unlock command.

    Unlocks DEMO access with unlock code.
    """
    user = update.effective_user

    # Check if code provided
    if not context.args or len(context.args) == 0:
        await update.message.reply_text(
            "⚠️ **Użycie:** /unlock <kod>\n\n"
            "Przykład: /unlock DEMO2024\n\n"
            "Nie masz kodu? Skontaktuj się z administratorem.",
            parse_mode="Markdown",
        )
        return

    unlock_code = context.args[0]

    backend = BackendClient()
    cache = UserCache()

    try:
        # Unlock via backend
        response = await backend.unlock_demo(user.id, unlock_code)

        # Invalidate cache
        await cache.set_user_data(user.id, response, ttl=3600)

        await update.message.reply_text(
            f"✅ **Dostęp DEMO odblokowany!**\n\n"
            f"Twoja rola: **{response['role']}**\n"
            f"Status: ✅ Autoryzowany\n\n"
            f"Możesz teraz korzystać z bota. Wyślij wiadomość aby zacząć!",
            parse_mode="Markdown",
        )

    except Exception as e:
        error_message = str(e)

        if "401" in error_message or "Nieprawidłowy" in error_message:
            await update.message.reply_text(
                "❌ **Nieprawidłowy kod odblokowania.**\n\n"
                "Sprawdź kod i spróbuj ponownie.",
                parse_mode="Markdown",
            )
        elif "404" in error_message:
            await update.message.reply_text(
                "❌ **Użytkownik nie istnieje.**\n\n"
                "Najpierw użyj /start aby się zarejestrować.",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                f"❌ Błąd: {error_message}\n\nSpróbuj ponownie."
            )

    finally:
        await backend.close()
        await cache.close()
