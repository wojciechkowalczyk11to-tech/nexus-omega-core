"""
Chat API routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.exceptions import AllProvidersFailedError, PolicyDeniedError
from app.core.logging_config import get_logger
from app.db.models.user import User
from app.services.orchestrator import Orchestrator, OrchestratorRequest

router = APIRouter()
logger = get_logger(__name__)


class ChatRequest(BaseModel):
    """Chat request schema."""

    query: str = Field(..., description="User query")
    session_id: int | None = Field(None, description="Optional session ID")
    mode: str | None = Field(None, description="Mode override (eco, smart, deep)")
    provider: str | None = Field(None, description="Provider override")
    deep_confirmed: bool = Field(False, description="DEEP mode confirmation")


class ChatResponse(BaseModel):
    """Chat response schema."""

    content: str
    provider: str
    model: str
    profile: str
    difficulty: str
    cost_usd: float
    latency_ms: int
    input_tokens: int
    output_tokens: int
    fallback_used: bool
    needs_confirmation: bool
    session_id: int | None
    sources: list[dict] | None = None
    meta_footer: str


class ProvidersResponse(BaseModel):
    """Providers list response."""

    providers: list[dict[str, str | bool]]


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Send a chat message and get AI response.

    Processes through 9-step orchestrator flow:
    1. Policy check
    2. Session management
    3. Context building
    4. Difficulty classification
    5. Profile selection
    6. Confirmation check
    7. Generation with fallback
    8. Persistence
    9. Snapshot creation
    """
    logger.info(f"Chat request from user {current_user.telegram_id}: {request.query[:50]}...")

    orchestrator = Orchestrator(db)

    try:
        orchestrator_request = OrchestratorRequest(
            user=current_user,
            query=request.query,
            session_id=request.session_id,
            mode_override=request.mode,
            provider_override=request.provider,
            deep_confirmed=request.deep_confirmed,
        )

        response = await orchestrator.process(orchestrator_request)

        # Build meta footer
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

        return ChatResponse(
            content=response.content,
            provider=response.provider,
            model=response.model,
            profile=response.profile,
            difficulty=response.difficulty,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            fallback_used=response.fallback_used,
            needs_confirmation=response.needs_confirmation,
            session_id=response.session_id,
            sources=response.sources,
            meta_footer=meta_footer,
        )

    except PolicyDeniedError as e:
        logger.warning(f"Policy denied for user {current_user.telegram_id}: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=e.message,
        )
    except AllProvidersFailedError as e:
        logger.error(f"All providers failed for user {current_user.telegram_id}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=e.message,
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd przetwarzania zapytania: {str(e)}",
        )


@router.get("/providers", response_model=ProvidersResponse)
async def get_providers(
    current_user: User = Depends(get_current_user),
) -> ProvidersResponse:
    """
    Get list of available AI providers for current user.
    """
    from app.services.policy_engine import PolicyEngine

    # Static provider list with role-based filtering
    all_providers = [
        {"name": "gemini", "display_name": "Google Gemini", "free": False},
        {"name": "deepseek", "display_name": "DeepSeek", "free": False},
        {"name": "groq", "display_name": "Groq", "free": True},
        {"name": "openrouter", "display_name": "OpenRouter", "free": True},
        {"name": "grok", "display_name": "xAI Grok", "free": False},
        {"name": "openai", "display_name": "OpenAI GPT-4", "free": False},
        {"name": "claude", "display_name": "Anthropic Claude", "free": False},
    ]

    # Filter by user role
    role = current_user.role
    provider_access = PolicyEngine.PROVIDER_ACCESS.get(role, {})

    available_providers = [
        {**p, "available": provider_access.get(p["name"], False)}
        for p in all_providers
    ]

    return ProvidersResponse(providers=available_providers)
