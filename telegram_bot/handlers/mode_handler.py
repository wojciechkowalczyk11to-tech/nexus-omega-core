"""
/mode command handler.
"""

from telegram import Update
from telegram.ext import ContextTypes

from telegram_bot.services.user_cache import UserCache


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /mode command.

    Allows user to change AI mode (eco, smart, deep).
    """
    user = update.effective_user
    cache = UserCache()

    try:
        # Get current mode
        current_mode = await cache.get_user_mode(user.id) or "eco"

        # Check if mode argument provided
        if context.args and len(context.args) > 0:
            new_mode = context.args[0].lower()

            if new_mode not in ("eco", "smart", "deep"):
                await update.message.reply_text("⚠️ Nieprawidłowy tryb. Dostępne: eco, smart, deep")
                return

            # Set new mode
            await cache.set_user_mode(user.id, new_mode)

            mode_descriptions = {
                "eco": "🌱 **ECO** - Szybki, ekonomiczny (Gemini 2.0 Flash, Groq)",
                "smart": "🧠 **SMART** - Zbalansowany (DeepSeek Reasoner, Gemini Thinking)",
                "deep": "🔬 **DEEP** - Zaawansowany (Gemini 2.5 Pro, GPT-4o, Claude) - wymaga FULL ACCESS",
            }

            await update.message.reply_text(
                f"✅ Zmieniono tryb na: {mode_descriptions[new_mode]}",
                parse_mode="Markdown",
            )

        else:
            # Show current mode and options
            mode_info = f"""🎛 **Aktualny tryb:** {current_mode.upper()}

**Dostępne tryby:**

🌱 **ECO** - Szybki, ekonomiczny
   Providery: Gemini 2.0 Flash, Groq, DeepSeek Chat
   Koszt: ~$0
   Użyj: `/mode eco`

🧠 **SMART** - Zbalansowany
   Providery: DeepSeek Reasoner, Gemini Thinking
   Koszt: ~$0.001-0.01 / zapytanie
   Użyj: `/mode smart`

🔬 **DEEP** - Zaawansowany (wymaga FULL ACCESS)
   Providery: DeepSeek, Gemini 2.5 Pro, GPT-4o, Claude Sonnet
   Koszt: ~$0.01-0.10 / zapytanie
   Użyj: `/mode deep`

💡 **Wskazówka:** Bot automatycznie wybiera tryb na podstawie trudności zapytania.
🔌 Użyj `/provider` aby wymusić konkretnego providera.
"""

            await update.message.reply_text(mode_info, parse_mode="Markdown")

    finally:
        await cache.close()
