"""
Document upload handler for RAG.
"""

from telegram import Update
from telegram.ext import ContextTypes

from telegram_bot.services.backend_client import BackendClient
from telegram_bot.services.user_cache import UserCache


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle document uploads.

    Uploads document to RAG system.
    """
    user = update.effective_user
    document = update.message.document

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

        # Check file size (max 20MB)
        max_size = 20 * 1024 * 1024
        if document.file_size > max_size:
            await update.message.reply_text(
                "⚠️ Plik jest za duży (max 20MB)."
            )
            return

        # Download file
        await update.message.reply_text("📥 Pobieranie pliku...")

        file = await document.get_file()
        file_bytes = await file.download_as_bytearray()

        # Upload to RAG
        await update.message.reply_text("⚙️ Przetwarzanie dokumentu...")

        response = await backend.upload_rag_document(
            token=token,
            filename=document.file_name,
            content=bytes(file_bytes),
        )

        # Success message
        await update.message.reply_text(
            f"✅ **{response['message']}**\n\n"
            f"ID dokumentu: {response['item_id']}\n"
            f"Fragmentów: {response['chunk_count']}\n\n"
            f"Możesz teraz zadawać pytania o zawartość tego dokumentu!",
            parse_mode="Markdown",
        )

    except Exception as e:
        error_message = str(e)

        if "403" in error_message:
            await update.message.reply_text(
                "⛔ **Upload dokumentów wymaga roli FULL_ACCESS.**\n\n"
                "Użyj /subscribe aby wykupić subskrypcję.",
                parse_mode="Markdown",
            )
        elif "400" in error_message:
            await update.message.reply_text(
                f"⚠️ Błąd przetwarzania pliku: {error_message}\n\n"
                f"Obsługiwane formaty: .txt, .md, .pdf, .docx, .html, .json"
            )
        else:
            await update.message.reply_text(
                f"❌ Błąd uploadu: {error_message}"
            )

    finally:
        await backend.close()
        await cache.close()
