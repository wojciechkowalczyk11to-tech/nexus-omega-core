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

**Zarządzanie kontem:**
/unlock <kod> - Odblokuj dostęp DEMO
/subscribe - Kup subskrypcję FULL_ACCESS
/usage - Statystyki użycia

**Sesje i pamięć:**
/session - Zarządzaj sesjami
/memory - Zarządzaj pamięcią absolutną
/export - Eksportuj konwersację

**Dokumenty (FULL_ACCESS):**
/rag - Zarządzaj dokumentami RAG
📎 Wyślij plik - Upload dokumentu

**Admin (tylko ADMIN):**
/admin - Panel administratora
/stats - Statystyki systemu
/invite - Generuj kod zaproszenia

**Tryby AI:**
🌱 **ECO** - Szybki, darmowy (Gemini, Groq)
🧠 **SMART** - Zbalansowany (DeepSeek Reasoner)
🔬 **DEEP** - Zaawansowany (GPT-4, Claude)

**Providery:**
- Google Gemini (Flash, Thinking, Exp)
- DeepSeek (Chat, Reasoner)
- Groq (Llama 3.3 70B)
- OpenRouter (Llama free tier)
- xAI Grok (Beta)
- OpenAI (GPT-4o)
- Anthropic Claude (Sonnet)

**Funkcje:**
✅ Multi-provider AI z fallback
✅ Baza wiedzy (Vertex AI Search)
✅ Dokumenty użytkownika (RAG)
✅ Wyszukiwanie w internecie
✅ Pamięć konwersacji
✅ Automatyczna klasyfikacja trudności
✅ Śledzenie kosztów

💬 Wyślij mi wiadomość, aby zacząć rozmowę!
"""

    await update.message.reply_text(help_text, parse_mode="Markdown")
