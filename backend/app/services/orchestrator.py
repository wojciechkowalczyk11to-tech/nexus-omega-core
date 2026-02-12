"""
Orchestrator for the 9-step AI request flow.

Flow:
1. Policy check (access control)
2. Get or create session
3. Build context (Vertex + vault + history)
4. Classify difficulty
5. Select profile
6. Check if DEEP needs confirmation
7. Generate with fallback chain
8. Persist (messages + ledger + counters)
9. Maybe create snapshot
"""

import time
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AllProvidersFailedError, PolicyDeniedError
from app.core.logging_config import get_logger
from app.db.models.user import User
from app.services.memory_manager import MemoryManager
from app.services.model_router import ModelRouter, Profile
from app.services.policy_engine import PolicyEngine
from app.services.usage_service import UsageService

logger = get_logger(__name__)


@dataclass
class OrchestratorRequest:
    """Request to the orchestrator."""

    user: User
    query: str
    session_id: int | None = None
    mode_override: str | None = None
    provider_override: str | None = None
    deep_confirmed: bool = False


@dataclass
class OrchestratorResponse:
    """Response from the orchestrator."""

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
    needs_confirmation: bool = False
    session_id: int | None = None
    sources: list[dict] | None = None


class Orchestrator:
    """Central orchestrator for AI request processing."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.policy_engine = PolicyEngine(db)
        self.memory_manager = MemoryManager(db)
        self.model_router = ModelRouter()
        self.usage_service = UsageService(db)

    async def process(self, request: OrchestratorRequest) -> OrchestratorResponse:
        """
        Process AI request through 9-step flow.

        Args:
            request: OrchestratorRequest

        Returns:
            OrchestratorResponse

        Raises:
            PolicyDeniedError: If access denied
            AllProvidersFailedError: If all providers fail
        """
        start_time = time.time()

        # Step 1: Policy check
        logger.info(f"Step 1: Policy check for user {request.user.telegram_id}")
        policy_result = await self.policy_engine.check_access(
            user=request.user,
            action="chat",
            provider=request.provider_override,
            profile=request.mode_override or request.user.default_mode,
        )

        if not policy_result.allowed:
            raise PolicyDeniedError(policy_result.reason)

        # Step 2: Get or create session
        logger.info("Step 2: Get or create session")
        session = await self.memory_manager.get_or_create_session(
            user_id=request.user.telegram_id,
            mode=request.mode_override or request.user.default_mode,
        )

        # Step 3: Build context
        logger.info("Step 3: Build context")
        
        from app.services.context_builder import ContextBuilder
        
        context_builder = ContextBuilder(self.db)
        context_messages, sources = await context_builder.build_context(
            user_id=request.user.telegram_id,
            session_id=session.id,
            query=request.query,
            use_vertex=True,
            use_rag=True,
            use_web=False,  # Web search disabled by default (costs credits)
        )

        # Step 4: Classify difficulty
        logger.info("Step 4: Classify difficulty")
        difficulty = self.model_router.classify_difficulty(request.query)

        # Step 5: Select profile
        logger.info("Step 5: Select profile")
        profile = self.model_router.select_profile(
            difficulty=difficulty,
            user_mode=request.mode_override,
            user_role=request.user.role,
        )

        # Step 6: Check if DEEP needs confirmation
        logger.info("Step 6: Check confirmation")
        if self.model_router.needs_confirmation(profile, request.user.role):
            if not request.deep_confirmed:
                return OrchestratorResponse(
                    content="",
                    provider="",
                    model="",
                    profile=profile.value,
                    difficulty=difficulty.value,
                    cost_usd=0.0,
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    fallback_used=False,
                    needs_confirmation=True,
                    session_id=session.id,
                )

        # Step 7: Generate with fallback chain
        logger.info("Step 7: Generate with fallback")
        
        # Import provider factory
        from app.providers.factory import ProviderFactory
        
        # Generate with fallback
        provider_response, provider_used, fallback_used = await ProviderFactory.generate_with_fallback(
            provider_chain=policy_result.provider_chain,
            messages=context_messages,
            profile=profile.value,
            temperature=0.7,
            max_tokens=2048,
        )
        
        response_content = provider_response.content
        model_used = provider_response.model
        input_tokens = provider_response.input_tokens
        output_tokens = provider_response.output_tokens
        cost_usd = provider_response.cost_usd

        # Step 8: Persist
        logger.info("Step 8: Persist messages and usage")

        # Persist user message
        await self.memory_manager.persist_message(
            session_id=session.id,
            user_id=request.user.telegram_id,
            role="user",
            content=request.query,
        )

        # Persist assistant message
        await self.memory_manager.persist_message(
            session_id=session.id,
            user_id=request.user.telegram_id,
            role="assistant",
            content=response_content,
            metadata={
                "provider": provider_used,
                "model": model_used,
                "profile": profile.value,
                "difficulty": difficulty.value,
                "cost_usd": cost_usd,
            },
        )

        # Log usage
        await self.usage_service.log_request(
            user_id=request.user.telegram_id,
            session_id=session.id,
            provider=provider_used,
            model=model_used,
            profile=profile.value,
            difficulty=difficulty.value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=int((time.time() - start_time) * 1000),
            fallback_used=fallback_used,
        )

        # Increment counters
        if profile == Profile.SMART:
            total_tokens = input_tokens + output_tokens
            smart_credits = self.model_router.calculate_smart_credits(total_tokens)
            await self.policy_engine.increment_counter(
                telegram_id=request.user.telegram_id,
                field="smart_credits_used",
                amount=smart_credits,
                cost_usd=cost_usd,
            )

        # Step 9: Maybe create snapshot
        logger.info("Step 9: Maybe create snapshot")
        await self.memory_manager.maybe_create_snapshot(session.id)

        # Commit all changes
        await self.db.commit()

        latency_ms = int((time.time() - start_time) * 1000)

        return OrchestratorResponse(
            content=response_content,
            provider=provider_used,
            model=model_used,
            profile=profile.value,
            difficulty=difficulty.value,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            fallback_used=fallback_used,
            needs_confirmation=False,
            session_id=session.id,
            sources=sources,
        )
