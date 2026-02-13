"""
Streaming chat API routes with Server-Sent Events (SSE).
"""

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.exceptions import AllProvidersFailedError, PolicyDeniedError
from app.core.logging_config import get_logger
from app.db.models.user import User
from app.services.orchestrator import Orchestrator, OrchestratorRequest

router = APIRouter()
logger = get_logger(__name__)


class ChatStreamRequest(BaseModel):
    """Chat stream request schema."""

    query: str = Field(..., description="User query")
    session_id: int | None = Field(None, description="Optional session ID")
    mode: str | None = Field(None, description="Mode override (eco, smart, deep)")
    provider: str | None = Field(None, description="Provider override")
    deep_confirmed: bool = Field(False, description="DEEP mode confirmation")


async def generate_stream_events(
    orchestrator_request: OrchestratorRequest,
    orchestrator: Orchestrator,
    current_user: User,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for streaming chat response.

    Yields:
        SSE formatted strings
    """
    try:
        # Send initial status event
        yield f"data: {json.dumps({'type': 'status', 'message': 'Processing request...'})}\n\n"

        # Process through orchestrator
        response = await orchestrator.process(orchestrator_request)

        # Check if needs confirmation
        if response.needs_confirmation:
            yield f"data: {json.dumps({'type': 'confirmation_needed', 'profile': response.profile})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Stream the response content in chunks
        # Since we already have the full response, we'll simulate streaming by chunking
        content = response.content
        chunk_size = 50  # characters per chunk

        for i in range(0, len(content), chunk_size):
            chunk = content[i : i + chunk_size]
            yield f"data: {json.dumps({'type': 'content', 'chunk': chunk})}\n\n"
            # Small delay to simulate streaming (optional)
            await asyncio.sleep(0.05)

        # Send metadata footer
        meta_footer = (
            f"🤖 {response.provider}-{response.model} | "
            f"💳 ${response.cost_usd:.4f} | "
            f"⚡ {response.input_tokens + response.output_tokens} tok | "
            f"⏱ {response.latency_ms / 1000:.1f}s"
        )

        if response.sources:
            source_list = " ".join(
                [f"{i+1}) {s['title']}" for i, s in enumerate(response.sources[:3])]
            )
            meta_footer += f"\n📚 Źródła (Vertex): {source_list}"

        yield f"data: {json.dumps({'type': 'metadata', 'footer': meta_footer, 'session_id': response.session_id, 'cost_usd': response.cost_usd, 'latency_ms': response.latency_ms})}\n\n"

        # Send completion event
        yield "data: [DONE]\n\n"

    except PolicyDeniedError as e:
        logger.warning(f"Policy denied for user {current_user.telegram_id}: {e.message}")
        yield f"data: {json.dumps({'type': 'error', 'message': e.message, 'code': 403})}\n\n"
        yield "data: [DONE]\n\n"

    except AllProvidersFailedError as e:
        logger.error(f"All providers failed for user {current_user.telegram_id}")
        yield f"data: {json.dumps({'type': 'error', 'message': e.message, 'code': 503})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Unexpected error in streaming chat: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': f'Błąd przetwarzania zapytania: {str(e)}', 'code': 500})}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/chat/stream")
async def chat_stream(
    request: ChatStreamRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Send a chat message and get streaming AI response via Server-Sent Events.

    Processes through 9-step orchestrator flow with real-time streaming.
    """
    logger.info(f"Streaming chat request from user {current_user.telegram_id}: {request.query[:50]}...")

    orchestrator = Orchestrator(db)

    orchestrator_request = OrchestratorRequest(
        user=current_user,
        query=request.query,
        session_id=request.session_id,
        mode_override=request.mode,
        provider_override=request.provider,
        deep_confirmed=request.deep_confirmed,
    )

    return StreamingResponse(
        generate_stream_events(orchestrator_request, orchestrator, current_user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
