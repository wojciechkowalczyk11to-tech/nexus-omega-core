"""
Chat message handler with streaming support.
"""

import asyncio
import json

import httpx
from telegram import Update
from telegram.ext import ContextTypes

from telegram_bot.config import settings
from telegram_bot.services.user_cache import UserCache


async def chat_message_streaming(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle chat messages with streaming response.

    Sends message to backend API and streams AI response in real-time
    by editing the Telegram message as chunks arrive.
    """
    user = update.effective_user
    query = update.message.text

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

        # Get user provider override
        provider = await cache.get_user_provider(user.id)

        # Send typing indicator
        await update.message.chat.send_action("typing")

        # Send initial message that will be updated
        response_message = await update.message.reply_text("⏳ Przetwarzam zapytanie...")

        # Stream response from backend
        accumulated_content = ""
        meta_footer = ""
        last_update_time = asyncio.get_event_loop().time()
        update_interval = 1.0  # Update message every 1 second

        async with (
            httpx.AsyncClient(timeout=120.0) as client,
            client.stream(
                "POST",
                f"{settings.backend_url}/api/v1/chat/stream",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "query": query,
                    "mode": mode,
                    "provider": provider,
                },
            ) as response,
        ):
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    event_type = data.get("type")

                    if event_type == "status":
                        # Update status message
                        await response_message.edit_text(f"⏳ {data['message']}")

                    elif event_type == "content":
                        # Accumulate content chunks
                        accumulated_content += data["chunk"]

                        # Update message periodically to avoid rate limits
                        current_time = asyncio.get_event_loop().time()
                        if current_time - last_update_time >= update_interval:
                            try:
                                # Truncate if too long (Telegram limit: 4096 chars)
                                display_content = accumulated_content
                                if len(display_content) > 3800:
                                    display_content = (
                                        display_content[:3800] + "\n\n... (kontynuacja)"
                                    )

                                await response_message.edit_text(
                                    display_content,
                                    parse_mode="Markdown",
                                )
                                last_update_time = current_time
                            except Exception:
                                # Ignore edit errors (message unchanged, rate limit, etc.)
                                pass

                    elif event_type == "confirmation_needed":
                        await response_message.edit_text(
                            "⚠️ **Tryb DEEP** wymaga potwierdzenia (wyższy koszt).\n\n"
                            "Użyj /deep_confirm aby potwierdzić, lub /mode eco aby zmienić tryb.",
                            parse_mode="Markdown",
                        )
                        # Store query in context for confirmation
                        context.user_data["pending_query"] = query
                        return

                    elif event_type == "metadata":
                        # Store metadata footer
                        meta_footer = data["footer"]

                    elif event_type == "error":
                        error_code = data.get("code", 500)
                        error_message = data["message"]

                        if error_code == 403:
                            await response_message.edit_text(
                                "⛔ Brak dostępu. Sprawdź swoją rolę: /help"
                            )
                        elif error_code == 503:
                            await response_message.edit_text(
                                "❌ Wszystkie providery AI są niedostępne. Spróbuj ponownie za chwilę."
                            )
                        else:
                            await response_message.edit_text(
                                f"❌ Błąd: {error_message}\n\nSpróbuj ponownie lub użyj /help"
                            )
                        return

                except json.JSONDecodeError:
                    # Skip malformed JSON
                    continue

        # Final update with complete content and metadata
        if accumulated_content:
            # Truncate if too long
            if len(accumulated_content) > 3800:
                accumulated_content = accumulated_content[:3800] + "\n\n... (odpowiedź skrócona)"

            full_response = f"{accumulated_content}\n\n---\n{meta_footer}"

            await response_message.edit_text(full_response, parse_mode="Markdown")

    except httpx.HTTPStatusError as e:
        error_message = f"HTTP {e.response.status_code}"
        if e.response.status_code == 403:
            await update.message.reply_text("⛔ Brak dostępu. Sprawdź swoją rolę: /help")
        elif e.response.status_code == 503:
            await update.message.reply_text(
                "❌ Wszystkie providery AI są niedostępne. Spróbuj ponownie za chwilę."
            )
        else:
            await update.message.reply_text(
                f"❌ Błąd: {error_message}\n\nSpróbuj ponownie lub użyj /help"
            )

    except Exception as e:
        await update.message.reply_text(f"❌ Błąd: {str(e)}\n\nSpróbuj ponownie lub użyj /help")

    finally:
        await cache.close()
