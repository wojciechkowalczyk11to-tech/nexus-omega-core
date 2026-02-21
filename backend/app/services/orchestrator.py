"""
ReAct Orchestrator — pętla rozumowania agenta AI.

Wzorzec: Reason → Act → Observe → Think → (loop) → Respond

Architektura:
    1. Policy check + session setup (jednorazowo)
    2. Budowanie kontekstu z priorytetami tokenów
    3. PĘTLA ReAct (max N iteracji):
       a) REASON  — LLM generuje myśl (thought) i decyzję o działaniu
       b) ACT     — wykonanie narzędzia lub wygenerowanie odpowiedzi
       c) OBSERVE — analiza wyniku narzędzia
       d) THINK   — self-correction: czy wynik jest poprawny? czy potrzeba innego podejścia?
    4. Finalizacja — persystencja, rozliczenie, snapshot

Self-correction:
    - Agent analizuje wyniki narzędzi pod kątem błędów
    - Jeśli narzędzie zwróci błąd → próbuje innego narzędzia lub podejścia
    - Jeśli wynik jest niewystarczający → może wywołać dodatkowe narzędzie
    - Maksymalna liczba iteracji zapobiega nieskończonym pętlom
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AllProvidersFailedError, PolicyDeniedError, ProviderError
from app.core.logging_config import get_logger
from app.db.models.user import User
from app.services.memory_manager import MemoryManager
from app.services.model_router import ModelRouter, Profile
from app.services.policy_engine import PolicyEngine
from app.services.token_budget_manager import (
    BudgetReport,
    MessagePriority,
    PrioritizedMessage,
    TokenBudgetManager,
    TokenCounter,
)
from app.services.usage_service import UsageService
from app.tools.tool_registry import ToolResult, create_default_tools

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REACT_ITERATIONS = 6  # Maximum tool-use loops
MAX_SELF_CORRECTIONS = 2  # Maximum self-correction attempts per iteration
TOOL_CALL_FINISH = "tool_calls"  # finish_reason indicating tool call


class AgentAction(str, Enum):
    """Possible actions in the ReAct loop."""

    THINK = "think"
    USE_TOOL = "use_tool"
    RESPOND = "respond"
    SELF_CORRECT = "self_correct"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ThoughtStep:
    """Single step in the agent's reasoning chain."""

    iteration: int
    action: AgentAction
    thought: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: ToolResult | None = None
    correction_reason: str | None = None
    timestamp_ms: int = 0


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
    # ReAct metadata
    reasoning_steps: list[ThoughtStep] = field(default_factory=list)
    react_iterations: int = 0
    tools_used: list[str] = field(default_factory=list)
    budget_report: BudgetReport | None = None
    trace_id: str = ""


# ---------------------------------------------------------------------------
# ReAct System Prompt
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """Jesteś NexusOmegaCore — zaawansowanym agentem AI z dostępem do narzędzi.

## Twoje zasady rozumowania (ReAct):
1. **MYŚL** (Thought): Zanim podejmiesz działanie, zawsze najpierw przemyśl problem.
2. **DZIAŁAJ** (Action): Jeśli potrzebujesz informacji — użyj odpowiedniego narzędzia.
3. **OBSERWUJ** (Observation): Przeanalizuj wynik narzędzia.
4. **OCEŃ** (Evaluate): Czy masz wystarczające informacje? Czy wynik jest poprawny?
5. **ODPOWIEDZ** (Respond): Gdy masz wszystkie potrzebne informacje — sformułuj odpowiedź.

## Zasady:
- Odpowiadaj po polsku, chyba że użytkownik poprosi o inny język
- Zawsze cytuj źródła, jeśli korzystasz z zewnętrznych informacji
- Bądź precyzyjny i konkretny
- Przyznaj się, jeśli czegoś nie wiesz
- Formatuj odpowiedzi czytelnie (Markdown)
- Jeśli narzędzie zwróci błąd, spróbuj innego podejścia
- NIE używaj narzędzi, jeśli możesz odpowiedzieć z własnej wiedzy
- Gdy użytkownik pyta o otwarty PR: doradź porównanie commitów/plików
- Sprawdź status CI/checks przed rekomendacją merge
- Nie doradzaj usuwania PR, jeśli zawiera unikalne zmiany
- Rekomenduj squash and merge dopiero gdy checks są zielone
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    ReAct Orchestrator — centralny silnik agenta AI.

    Implementuje pętlę Reason-Act-Observe-Think z:
    - Natywnym function calling (OpenAI/Gemini/Claude)
    - Self-correction (automatyczna naprawa błędów)
    - Token Budget Management (inteligentne przycinanie kontekstu)
    - Pełnym śledzeniem rozumowania (reasoning trace)
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.policy_engine = PolicyEngine(db)
        self.memory_manager = MemoryManager(db)
        self.model_router = ModelRouter()
        self.usage_service = UsageService(db)
        self.tool_registry = create_default_tools(db)

    async def process(self, request: OrchestratorRequest) -> OrchestratorResponse:
        """
        Process AI request through ReAct loop.

        Flow:
        1. Policy check + session setup
        2. Build prioritized context with token budgeting
        3. ReAct loop (Reason → Act → Observe → Think)
        4. Persist results and return response

        Args:
            request: OrchestratorRequest

        Returns:
            OrchestratorResponse with content and reasoning trace

        Raises:
            PolicyDeniedError: If access denied
            AllProvidersFailedError: If all providers fail
        """
        trace_id = str(uuid.uuid4())[:12]
        start_time = time.time()
        reasoning_steps: list[ThoughtStep] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost_usd = 0.0
        tools_used: list[str] = []
        fallback_used = False

        # =====================================================================
        # STEP 1: Policy check
        # =====================================================================
        logger.info(f"[{trace_id}] Step 1: Policy check for user {request.user.telegram_id}")
        policy_result = await self.policy_engine.check_access(
            user=request.user,
            action="chat",
            provider=request.provider_override,
            profile=request.mode_override or request.user.default_mode,
        )

        if not policy_result.allowed:
            raise PolicyDeniedError(policy_result.reason)

        # =====================================================================
        # STEP 2: Session management
        # =====================================================================
        logger.info(f"[{trace_id}] Step 2: Get or create session")
        session = await self.memory_manager.get_or_create_session(
            user_id=request.user.telegram_id,
            mode=request.mode_override or request.user.default_mode,
        )

        # =====================================================================
        # STEP 3: Classify difficulty & select profile
        # =====================================================================
        logger.info(f"[{trace_id}] Step 3: Classify difficulty & select profile")
        difficulty = self.model_router.classify_difficulty(request.query)
        profile = self.model_router.select_profile(
            difficulty=difficulty,
            user_mode=request.mode_override,
            user_role=request.user.role,
        )

        # Check DEEP confirmation
        if (
            self.model_router.needs_confirmation(profile, request.user.role)
            and not request.deep_confirmed
        ):
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
                trace_id=trace_id,
            )

        # =====================================================================
        # STEP 4: Build initial context with token budgeting
        # =====================================================================
        logger.info(f"[{trace_id}] Step 4: Build context with token budget")

        # Determine provider and model for budget calculation
        provider_chain = policy_result.provider_chain or ["gemini"]
        primary_provider = provider_chain[0]

        from app.providers.factory import ProviderFactory

        try:
            provider_instance = ProviderFactory.create(primary_provider)
            model_name = provider_instance.get_model_for_profile(profile.value)
        except ProviderError:
            model_name = ""

        # Initialize token budget manager
        budget_manager = TokenBudgetManager(
            model=model_name,
            provider=primary_provider,
        )

        # Build prioritized context
        prioritized_messages, sources = await self._build_prioritized_context(
            user_id=request.user.telegram_id,
            session_id=session.id,
            query=request.query,
        )

        # Apply token budget
        context_messages, budget_report = budget_manager.apply_budget(prioritized_messages)

        # =====================================================================
        # STEP 5: ReAct Loop
        # =====================================================================
        logger.info(f"[{trace_id}] Step 5: Starting ReAct loop")

        response_content = ""
        provider_used = ""
        model_used = ""

        for iteration in range(1, MAX_REACT_ITERATIONS + 1):
            logger.info(f"[{trace_id}] ReAct iteration {iteration}/{MAX_REACT_ITERATIONS}")

            step_start = time.time()

            # --- REASON + ACT: Call LLM with tools ---
            try:
                llm_response, p_used, fb_used = await self._call_llm_with_tools(
                    provider_chain=provider_chain,
                    messages=context_messages,
                    profile=profile.value,
                    provider_name=primary_provider,
                )
            except AllProvidersFailedError:
                # If all providers fail, try without tools as last resort
                logger.warning(f"[{trace_id}] All providers failed with tools, trying without")
                try:
                    llm_response, p_used, fb_used = await ProviderFactory.generate_with_fallback(
                        provider_chain=provider_chain,
                        messages=context_messages,
                        profile=profile.value,
                        temperature=0.7,
                        max_tokens=2048,
                    )
                except AllProvidersFailedError:
                    raise

            provider_used = p_used
            model_used = llm_response.model
            if fb_used:
                fallback_used = True

            total_input_tokens += llm_response.input_tokens
            total_output_tokens += llm_response.output_tokens
            total_cost_usd += llm_response.cost_usd

            # --- Check if LLM wants to use a tool ---
            tool_calls = self._extract_tool_calls(llm_response)

            if not tool_calls:
                # No tool calls — LLM is responding directly
                response_content = llm_response.content
                reasoning_steps.append(
                    ThoughtStep(
                        iteration=iteration,
                        action=AgentAction.RESPOND,
                        thought="Odpowiedź bezpośrednia — brak potrzeby użycia narzędzi.",
                        timestamp_ms=int((time.time() - step_start) * 1000),
                    )
                )
                logger.info(f"[{trace_id}] LLM responded directly (no tool calls)")
                break

            # --- Process tool calls ---
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                tc.get("id", "")

                logger.info(
                    f"[{trace_id}] Tool call: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})"
                )

                # Record thought step
                thought_step = ThoughtStep(
                    iteration=iteration,
                    action=AgentAction.USE_TOOL,
                    thought=f"Potrzebuję użyć narzędzia '{tool_name}' aby uzyskać informacje.",
                    tool_name=tool_name,
                    tool_args=tool_args,
                )

                # Execute tool
                tool_result = await self.tool_registry.execute(
                    tool_name=tool_name,
                    arguments=tool_args,
                    user_id=request.user.telegram_id,
                    db_session=self.db,
                )

                thought_step.tool_result = tool_result
                thought_step.timestamp_ms = int((time.time() - step_start) * 1000)
                reasoning_steps.append(thought_step)

                if tool_name not in tools_used:
                    tools_used.append(tool_name)

                # --- OBSERVE: Add tool result to context ---
                result_content = tool_result.to_message_content()

                # Add assistant's tool call message
                context_messages.append(
                    {
                        "role": "assistant",
                        "content": f"[Wywołuję narzędzie: {tool_name}]",
                    }
                )

                # Add tool result as observation
                if tool_result.success:
                    context_messages.append(
                        {
                            "role": "system",
                            "content": f"[Wynik narzędzia {tool_name}]:\n{result_content}",
                        }
                    )
                else:
                    # --- SELF-CORRECTION: Tool failed ---
                    context_messages.append(
                        {
                            "role": "system",
                            "content": (
                                f"[BŁĄD narzędzia {tool_name}]: {result_content}\n"
                                f"Spróbuj innego podejścia lub odpowiedz na podstawie dostępnej wiedzy."
                            ),
                        }
                    )

                    reasoning_steps.append(
                        ThoughtStep(
                            iteration=iteration,
                            action=AgentAction.SELF_CORRECT,
                            thought=f"Narzędzie '{tool_name}' zwróciło błąd. Szukam alternatywnego podejścia.",
                            correction_reason=tool_result.error,
                            timestamp_ms=int((time.time() - step_start) * 1000),
                        )
                    )

                    logger.warning(
                        f"[{trace_id}] Self-correction triggered: tool '{tool_name}' failed. "
                        f"Error: {tool_result.error}"
                    )

            # Re-apply token budget after adding tool results
            # (context may have grown significantly)
            current_tokens = TokenCounter.count_messages(context_messages)
            if current_tokens > budget_manager.effective_budget:
                logger.info(f"[{trace_id}] Re-applying token budget after tool results")
                # Rebuild prioritized messages from current context
                reprioritized = []
                for i, msg in enumerate(context_messages):
                    content = msg.get("content", "")
                    if msg["role"] == "system" and i == 0:
                        priority = MessagePriority.SYSTEM_PROMPT
                        truncatable = False
                    elif "[Wynik narzędzia" in content or "[BŁĄD narzędzia" in content:
                        priority = MessagePriority.TOOL_RESULT
                        truncatable = True
                    elif msg["role"] == "user" and i == len(context_messages) - 1:
                        priority = MessagePriority.CURRENT_QUERY
                        truncatable = False
                    elif msg["role"] == "user" or msg["role"] == "assistant":
                        priority = MessagePriority.HISTORY_RECENT
                        truncatable = True
                    else:
                        priority = MessagePriority.SYSTEM_CONTEXT
                        truncatable = True

                    reprioritized.append(
                        PrioritizedMessage(
                            message=msg,
                            priority=priority,
                            truncatable=truncatable,
                            source=f"react_iter_{iteration}",
                        )
                    )

                context_messages, budget_report = budget_manager.apply_budget(reprioritized)

        else:
            # Max iterations reached without final response
            logger.warning(f"[{trace_id}] Max ReAct iterations reached ({MAX_REACT_ITERATIONS})")
            if not response_content:
                # Force a final response
                context_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "[INSTRUKCJA SYSTEMOWA]: Osiągnięto maksymalną liczbę iteracji. "
                            "Sformułuj najlepszą możliwą odpowiedź na podstawie zebranych informacji."
                        ),
                    }
                )
                try:
                    final_response, p_used, _ = await ProviderFactory.generate_with_fallback(
                        provider_chain=provider_chain,
                        messages=context_messages,
                        profile=profile.value,
                        temperature=0.7,
                        max_tokens=2048,
                    )
                    response_content = final_response.content
                    total_input_tokens += final_response.input_tokens
                    total_output_tokens += final_response.output_tokens
                    total_cost_usd += final_response.cost_usd
                    provider_used = p_used
                    model_used = final_response.model
                except AllProvidersFailedError:
                    response_content = (
                        "Przepraszam, nie udało mi się wygenerować odpowiedzi. "
                        "Spróbuj ponownie lub uprość pytanie."
                    )

        # =====================================================================
        # STEP 6: Persist messages and usage
        # =====================================================================
        logger.info(f"[{trace_id}] Step 6: Persist messages and usage")

        # Persist user message
        await self.memory_manager.persist_message(
            session_id=session.id,
            user_id=request.user.telegram_id,
            role="user",
            content=request.query,
        )

        # Persist assistant message with reasoning metadata
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
                "cost_usd": total_cost_usd,
                "react_iterations": iteration if "iteration" in dir() else 0,
                "tools_used": tools_used,
                "trace_id": trace_id,
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
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=total_cost_usd,
            latency_ms=int((time.time() - start_time) * 1000),
            fallback_used=fallback_used,
        )

        # Increment counters
        if profile == Profile.SMART:
            total_tokens = total_input_tokens + total_output_tokens
            smart_credits = self.model_router.calculate_smart_credits(total_tokens)
            await self.policy_engine.increment_counter(
                telegram_id=request.user.telegram_id,
                field="smart_credits_used",
                amount=smart_credits,
                cost_usd=total_cost_usd,
            )

        # =====================================================================
        # STEP 7: Maybe create snapshot
        # =====================================================================
        logger.info(f"[{trace_id}] Step 7: Maybe create snapshot")
        await self.memory_manager.maybe_create_snapshot(session.id)

        # Commit all changes
        await self.db.commit()

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[{trace_id}] ReAct complete: {len(reasoning_steps)} steps, "
            f"{len(tools_used)} tools, {latency_ms}ms, ${total_cost_usd:.6f}"
        )

        return OrchestratorResponse(
            content=response_content,
            provider=provider_used,
            model=model_used,
            profile=profile.value,
            difficulty=difficulty.value,
            cost_usd=total_cost_usd,
            latency_ms=latency_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            fallback_used=fallback_used,
            needs_confirmation=False,
            session_id=session.id,
            sources=sources if "sources" in dir() else None,
            reasoning_steps=reasoning_steps,
            react_iterations=iteration if "iteration" in dir() else 0,
            tools_used=tools_used,
            budget_report=budget_report if "budget_report" in dir() else None,
            trace_id=trace_id,
        )

    # =========================================================================
    # Private methods
    # =========================================================================

    async def _build_prioritized_context(
        self,
        user_id: int,
        session_id: int,
        query: str,
    ) -> tuple[list[PrioritizedMessage], list[dict[str, Any]]]:
        """
        Build context messages with priority metadata for token budgeting.

        Returns:
            Tuple of (prioritized messages, sources)
        """
        prioritized: list[PrioritizedMessage] = []
        sources: list[dict[str, Any]] = []

        # 1. System prompt (highest priority, never truncated)
        system_prompt = REACT_SYSTEM_PROMPT

        # 2. Add user memory to system prompt
        memories = await self.memory_manager.list_absolute_memories(user_id)
        if memories:
            memory_text = "\n".join([f"- {mem.key}: {mem.value}" for mem in memories[:5]])
            system_prompt += f"\n\n**Preferencje użytkownika:**\n{memory_text}"

        # 3. Add available tools description to system prompt
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        system_prompt += f"\n\n**Dostępne narzędzia:**\n{tool_descriptions}"

        prioritized.append(
            PrioritizedMessage(
                message={"role": "system", "content": system_prompt},
                priority=MessagePriority.SYSTEM_PROMPT,
                truncatable=False,
                source="system_prompt",
            )
        )

        # 4. Session history (snapshot + recent messages)
        history_messages = await self.memory_manager.get_context_messages(session_id)
        for i, msg in enumerate(history_messages):
            content = msg.get("content", "")
            is_snapshot = "[Podsumowanie poprzedniej konwersacji]" in content

            if is_snapshot:
                prioritized.append(
                    PrioritizedMessage(
                        message=msg,
                        priority=MessagePriority.SNAPSHOT,
                        truncatable=True,
                        min_tokens=100,
                        source="snapshot",
                    )
                )
            else:
                # More recent messages get higher priority
                recency_boost = min(i * 2, 10)
                prioritized.append(
                    PrioritizedMessage(
                        message=msg,
                        priority=MessagePriority.HISTORY_OLD + recency_boost,
                        truncatable=True,
                        min_tokens=30,
                        source=f"history_{i}",
                    )
                )

        # 5. Current user query (highest priority, never truncated)
        prioritized.append(
            PrioritizedMessage(
                message={"role": "user", "content": query},
                priority=MessagePriority.CURRENT_QUERY,
                truncatable=False,
                source="current_query",
            )
        )

        return prioritized, sources

    async def _call_llm_with_tools(
        self,
        provider_chain: list[str],
        messages: list[dict[str, str]],
        profile: str,
        provider_name: str = "",
    ) -> tuple[Any, str, bool]:
        """
        Call LLM with native function calling (tool use).

        Attempts to use provider-native tool calling.
        Falls back to standard generation if tool calling is not supported.

        Returns:
            Tuple of (ProviderResponse, provider_name, fallback_used)
        """
        from app.providers.factory import ProviderFactory

        last_error = None
        fallback_used = False

        for i, prov_name in enumerate(provider_chain):
            try:
                logger.info(
                    f"Trying provider with tools: {prov_name} (attempt {i + 1}/{len(provider_chain)})"
                )

                provider = ProviderFactory.create(prov_name)
                model = provider.get_model_for_profile(profile)

                # Get tool schemas for this provider
                tool_schemas = self.tool_registry.get_tools_for_provider(prov_name)

                # Try native tool calling
                response = await self._generate_with_tools(
                    provider=provider,
                    provider_name=prov_name,
                    model=model,
                    messages=messages,
                    tools=tool_schemas,
                )

                if i > 0:
                    fallback_used = True

                return response, prov_name, fallback_used

            except ProviderError as e:
                logger.warning(f"Provider {prov_name} failed: {e.message}")
                last_error = e
                continue
            except Exception as e:
                logger.warning(f"Provider {prov_name} unexpected error: {e}")
                last_error = ProviderError(str(e), {"provider": prov_name})
                continue

        # All providers failed
        attempts = [
            {"provider": p, "error": str(last_error) if last_error else "Unknown error"}
            for p in provider_chain
        ]
        raise AllProvidersFailedError(attempts=attempts)

    async def _generate_with_tools(
        self,
        provider: Any,
        provider_name: str,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """
        Generate completion with tool calling support.

        Handles provider-specific tool calling APIs:
        - OpenAI/DeepSeek/Groq/Grok: tools parameter
        - Claude: tools parameter with different schema
        - Gemini: function_declarations

        Falls back to standard generation if tools are empty or unsupported.
        """

        if not tools:
            # No tools available — standard generation
            return await provider.generate(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=2048,
            )

        # Provider-specific tool calling
        if provider_name in ("openai", "deepseek", "groq", "grok", "openrouter"):
            return await self._openai_style_tool_call(provider, model, messages, tools)
        elif provider_name == "claude":
            return await self._claude_tool_call(provider, model, messages, tools)
        elif provider_name == "gemini":
            return await self._gemini_tool_call(provider, model, messages, tools)
        else:
            # Unknown provider — standard generation
            return await provider.generate(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=2048,
            )

    async def _openai_style_tool_call(
        self,
        provider: Any,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Handle OpenAI-style tool calling (OpenAI, DeepSeek, Groq, Grok)."""
        from app.providers.base import ProviderResponse

        start_time = time.time()

        try:
            # Use the provider's client directly for tool calling
            if not hasattr(provider, "client") or provider.client is None:
                return await provider.generate(
                    messages=messages,
                    model=model,
                    temperature=0.7,
                    max_tokens=2048,
                )

            response = await provider.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=2048,
            )

            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost_usd = provider.calculate_cost(model, input_tokens, output_tokens)

            # Build response with tool call info in raw_response
            raw = {"finish_reason": finish_reason}
            if choice.message.tool_calls:
                raw["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {},
                    }
                    for tc in choice.message.tool_calls
                ]

            return ProviderResponse(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=int((time.time() - start_time) * 1000),
                finish_reason=finish_reason,
                raw_response=raw,
            )

        except Exception as e:
            logger.warning(f"OpenAI-style tool call failed, falling back to standard: {e}")
            return await provider.generate(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=2048,
            )

    async def _claude_tool_call(
        self,
        provider: Any,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Handle Claude (Anthropic) tool calling."""
        from app.providers.base import ProviderResponse

        start_time = time.time()

        try:
            if not hasattr(provider, "client") or provider.client is None:
                return await provider.generate(
                    messages=messages,
                    model=model,
                    temperature=0.7,
                    max_tokens=2048,
                )

            # Extract system message for Claude
            system_message = None
            claude_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    if system_message is None:
                        system_message = msg["content"]
                    else:
                        system_message += "\n\n" + msg["content"]
                else:
                    claude_messages.append(msg)

            request_params = {
                "model": model,
                "messages": claude_messages,
                "tools": tools,
                "temperature": 0.7,
                "max_tokens": 2048,
            }
            if system_message:
                request_params["system"] = system_message

            response = await provider.client.messages.create(**request_params)

            # Parse Claude response
            content = ""
            tool_calls_data = []

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls_data.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "arguments": block.input,
                        }
                    )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost_usd = provider.calculate_cost(model, input_tokens, output_tokens)

            raw = {"finish_reason": response.stop_reason}
            if tool_calls_data:
                raw["tool_calls"] = tool_calls_data

            return ProviderResponse(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=int((time.time() - start_time) * 1000),
                finish_reason=response.stop_reason,
                raw_response=raw,
            )

        except Exception as e:
            logger.warning(f"Claude tool call failed, falling back to standard: {e}")
            return await provider.generate(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=2048,
            )

    async def _gemini_tool_call(
        self,
        provider: Any,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Handle Gemini function calling."""
        import asyncio

        from app.providers.base import ProviderResponse

        start_time = time.time()

        try:
            import google.generativeai as genai

            # Convert messages to Gemini format
            gemini_messages = provider._convert_messages(messages)

            # Create model with tools
            gemini_tools = genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                pname: genai.protos.Schema(
                                    type=getattr(
                                        genai.protos.Type,
                                        pdef.get("type", "STRING"),
                                        genai.protos.Type.STRING,
                                    ),
                                    description=pdef.get("description", ""),
                                )
                                for pname, pdef in t.get("parameters", {})
                                .get("properties", {})
                                .items()
                            },
                            required=t.get("parameters", {}).get("required", []),
                        ),
                    )
                    for t in tools
                ]
            )

            model_instance = genai.GenerativeModel(
                model,
                tools=[gemini_tools],
            )

            generation_config = genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model_instance.generate_content(
                    gemini_messages,
                    generation_config=generation_config,
                ),
            )

            # Parse Gemini response
            content = ""
            tool_calls_data = []

            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        content += part.text
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        tool_calls_data.append(
                            {
                                "id": f"gemini_{fc.name}",
                                "name": fc.name,
                                "arguments": dict(fc.args) if fc.args else {},
                            }
                        )

            # Token counts
            try:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
            except (AttributeError, KeyError):
                input_tokens = sum(len(m.get("content", "").split()) * 2 for m in messages)
                output_tokens = len(content.split()) * 2

            cost_usd = provider.calculate_cost(model, input_tokens, output_tokens)

            raw = {"finish_reason": "stop"}
            if tool_calls_data:
                raw["tool_calls"] = tool_calls_data
                raw["finish_reason"] = "tool_calls"

            return ProviderResponse(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=int((time.time() - start_time) * 1000),
                finish_reason=raw["finish_reason"],
                raw_response=raw,
            )

        except Exception as e:
            logger.warning(f"Gemini tool call failed, falling back to standard: {e}")
            return await provider.generate(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=2048,
            )

    def _extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """
        Extract tool calls from provider response.

        Handles different response formats from various providers.

        Returns:
            List of tool call dicts with 'name', 'arguments', 'id'
        """
        if not response.raw_response:
            return []

        raw = response.raw_response

        # Check for tool_calls in raw response
        tool_calls = raw.get("tool_calls", [])
        if tool_calls:
            return tool_calls

        # Check finish_reason
        finish_reason = raw.get("finish_reason", response.finish_reason)
        if finish_reason in ("tool_calls", "tool_use", "function_call"):
            # Tool call indicated but not in expected format
            logger.warning("Tool call finish_reason but no tool_calls in response")

        return []
