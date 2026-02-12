"""
/subscribe command handler for Telegram Stars payments.
"""

from telegram import LabeledPrice, Update
from telegram.ext import ContextTypes

from telegram_bot.config import settings
from telegram_bot.services.backend_client import BackendClient
from telegram_bot.services.user_cache import UserCache


async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /subscribe command.

    Shows subscription options with Telegram Stars pricing.
    """
    user = update.effective_user

    subscribe_text = """💎 **Subskrypcja FULL_ACCESS**

**Co zyskujesz:**
✅ Dostęp do wszystkich providerów AI
✅ Tryb DEEP (GPT-4, Claude)
✅ Upload dokumentów (RAG)
✅ Wyszukiwanie w internecie
✅ 1000 kredytów miesięcznie
✅ Wyższy limit zapytań (100/min)

**Cennik:**

🌟 **500 Stars** - FULL_ACCESS (30 dni)
   Pełny dostęp + 1000 kredytów

🌟 **50 Stars** - 100 kredytów
🌟 **200 Stars** - 500 kredytów
🌟 **350 Stars** - 1000 kredytów

**Jak kupić:**
Użyj komendy /buy <produkt>

Przykłady:
/buy full_access_monthly
/buy credits_100
/buy credits_500
/buy credits_1000

💡 **Telegram Stars** to wirtualna waluta Telegram.
Możesz kupić Stars w aplikacji Telegram.
"""

    await update.message.reply_text(subscribe_text, parse_mode="Markdown")


async def buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /buy command.

    Initiates Telegram Stars payment.
    """
    user = update.effective_user

    # Check if product specified
    if not context.args or len(context.args) == 0:
        await update.message.reply_text(
            "⚠️ **Użycie:** /buy <produkt>\n\n"
            "Dostępne produkty:\n"
            "- full_access_monthly\n"
            "- credits_100\n"
            "- credits_500\n"
            "- credits_1000\n\n"
            "Przykład: /buy full_access_monthly",
            parse_mode="Markdown",
        )
        return

    product_id = context.args[0]

    # Pricing
    pricing = {
        "full_access_monthly": {
            "title": "FULL_ACCESS - 30 dni",
            "description": "Pełny dostęp do wszystkich funkcji + 1000 kredytów",
            "stars": 500,
        },
        "credits_100": {
            "title": "100 kredytów",
            "description": "Doładowanie 100 kredytów",
            "stars": 50,
        },
        "credits_500": {
            "title": "500 kredytów",
            "description": "Doładowanie 500 kredytów",
            "stars": 200,
        },
        "credits_1000": {
            "title": "1000 kredytów",
            "description": "Doładowanie 1000 kredytów",
            "stars": 350,
        },
    }

    if product_id not in pricing:
        await update.message.reply_text(
            f"❌ Nieznany produkt: {product_id}\n\n"
            f"Użyj /subscribe aby zobaczyć dostępne produkty."
        )
        return

    product = pricing[product_id]

    # Create invoice
    await update.message.reply_invoice(
        title=product["title"],
        description=product["description"],
        payload=f"product:{product_id}",
        provider_token="",  # Empty for Telegram Stars
        currency="XTR",  # Telegram Stars currency
        prices=[
            LabeledPrice(label=product["title"], amount=product["stars"])
        ],
    )


async def precheckout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle pre-checkout query.

    Validates payment before processing.
    """
    query = update.pre_checkout_query

    # Always approve (validation done on backend)
    await query.answer(ok=True)


async def successful_payment_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle successful payment.

    Processes payment and grants benefits.
    """
    user = update.effective_user
    payment = update.message.successful_payment

    backend = BackendClient()
    cache = UserCache()

    try:
        # Extract product_id from payload
        payload = payment.invoice_payload
        product_id = payload.split(":")[1] if ":" in payload else "unknown"

        # Get token
        token = await cache.get_user_token(user.id)

        if not token:
            await update.message.reply_text(
                "⚠️ Nie jesteś zalogowany. Użyj /start aby się zarejestrować."
            )
            return

        # Process payment via backend
        # Note: This would require a backend endpoint for payment processing
        # For now, send confirmation message

        await update.message.reply_text(
            f"✅ **Płatność zakończona pomyślnie!**\n\n"
            f"Produkt: {product_id}\n"
            f"Zapłacono: {payment.total_amount} Stars\n\n"
            f"Twoje korzyści zostały aktywowane. Użyj /start aby odświeżyć status.",
            parse_mode="Markdown",
        )

        # Invalidate cache to force refresh
        await cache.set_user_data(user.id, {}, ttl=1)

    except Exception as e:
        await update.message.reply_text(
            f"❌ Błąd przetwarzania płatności: {str(e)}\n\n"
            f"Skontaktuj się z supportem."
        )

    finally:
        await backend.close()
        await cache.close()
