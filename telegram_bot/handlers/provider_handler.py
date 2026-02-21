"""
/provider command handler — select a specific AI provider or return to auto-routing.
"""

from telegram import Update
from telegram.ext import ContextTypes

from telegram_bot.services.user_cache import UserCache

AVAILABLE_PROVIDERS = {
    "gemini": "Google Gemini (Flash / Thinking / 2.5 Pro)",
    "deepseek": "DeepSeek (Chat / Reasoner)",
    "groq": "Groq (Llama 3.3 70B)",
    "openrouter": "OpenRouter (Llama free tier)",
    "grok": "xAI Grok (Beta)",
    "openai": "OpenAI (GPT-4o)",
    "claude": "Anthropic Claude (Sonnet)",
}

PROVIDER_ALIASES = {
    "xai": "grok",
    "x.ai": "grok",
    "google": "gemini",
    "anthropic": "claude",
    "llama": "groq",
}


async def provider_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /provider command.

    Usage:
        /provider          - show current provider and available options
        /provider grok     - force all requests through Grok
        /provider claude   - force all requests through Claude
        /provider auto     - return to automatic routing
    """
    user = update.effective_user
    cache = UserCache()

    try:
        current_provider = await cache.get_user_provider(user.id)

        if context.args and len(context.args) > 0:
            choice = context.args[0].lower()

            # Handle "auto" / "reset" / "default"
            if choice in ("auto", "reset", "default"):
                await cache.set_user_provider(user.id, None)
                await update.message.reply_text(
                    "✅ Provider: **auto** — system automatycznie wybiera najlepszego providera.",
                    parse_mode="Markdown",
                )
                return

            # Resolve aliases
            resolved = PROVIDER_ALIASES.get(choice, choice)

            if resolved not in AVAILABLE_PROVIDERS:
                names = ", ".join(sorted(AVAILABLE_PROVIDERS.keys()))
                await update.message.reply_text(
                    f"⚠️ Nieznany provider: `{choice}`\n\n"
                    f"Dostępni: {names}\n"
                    f"Użyj `/provider auto` aby wrócić do automatycznego routingu.",
                    parse_mode="Markdown",
                )
                return

            await cache.set_user_provider(user.id, resolved)
            desc = AVAILABLE_PROVIDERS[resolved]
            await update.message.reply_text(
                f"✅ Provider ustawiony: **{resolved}** ({desc})\n\n"
                f"Wszystkie zapytania będą kierowane do tego providera.\n"
                f"Aby wrócić do auto-routingu: `/provider auto`",
                parse_mode="Markdown",
            )

        else:
            # Show current provider and options
            current_display = current_provider or "auto (automatyczny)"
            lines = [f"🔌 **Aktualny provider:** {current_display}\n"]
            lines.append("**Dostępni providerzy:**\n")
            for name, desc in AVAILABLE_PROVIDERS.items():
                marker = " 👈" if name == current_provider else ""
                lines.append(f"  `{name}` — {desc}{marker}")
            lines.append(f"\n**Użycie:**")
            lines.append("`/provider grok` — wymuszenie Grok")
            lines.append("`/provider claude` — wymuszenie Claude")
            lines.append("`/provider auto` — powrót do automatycznego routingu")

            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    finally:
        await cache.close()
