"""
/help command handler.
"""

from telegram import Update
from telegram.ext import ContextTypes


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /help command.

    Shows list of available commands.
    """
    help_text = """📚 **NexusOmegaCore - Pomoc**

**Podstawowe komendy:**
/start - Rozpocznij rozmowę
/help - Ta wiadomość
/mode - Zmień tryb AI (eco/smart/deep)
/provider - Wybierz providera AI (grok/claude/auto)

**Zarządzanie kontem:**
/unlock <kod> - Odblokuj dostęp DEMO
/subscribe - Kup subskrypcję FULL ACCESS
/buy - Kup kredyty (Telegram Stars)

**Dokumenty (FULL ACCESS):**
📎 Wyślij plik - Upload dokumentu do RAG

**Tryby AI:**
🌱 **ECO** - Szybki, ekonomiczny (Gemini Flash, Groq)
🧠 **SMART** - Zbalansowany (DeepSeek Reasoner, Gemini Thinking)
🔬 **DEEP** - Zaawansowany (Gemini 2.5 Pro, GPT-4o, Claude)

**Providery (wybierz przez /provider):**
- Google Gemini (2.0 Flash, Thinking, 2.5 Pro)
- DeepSeek (Chat, Reasoner)
- Groq (Llama 3.3 70B)
- OpenRouter (Llama free tier)
- xAI Grok (Beta)
- OpenAI (GPT-4o)
- Anthropic Claude (Sonnet)

**Funkcje:**
✅ Multi-provider AI z automatycznym fallback
✅ Wyszukiwanie w internecie (Brave Search)
✅ Dokumenty użytkownika (RAG)
✅ Pamięć konwersacji i preferencji
✅ Automatyczna klasyfikacja trudności
✅ Śledzenie kosztów
✅ ReAct agent z narzędziami

💬 Wyślij mi wiadomość, aby zacząć rozmowę!
"""

    await update.message.reply_text(help_text, parse_mode="Markdown")
