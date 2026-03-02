"""
Token Budget Manager — inteligentne zarządzanie budżetem tokenów.

Odpowiada za:
- Liczenie tokenów przed wysłaniem zapytania do LLM
- Inteligentne przycinanie kontekstu (smart truncation)
- Priorytetyzację informacji w kontekście
- Optymalizację kosztów

Strategia priorytetów (od najwyższego):
1. System prompt (nigdy nie przycinany)
2. Bieżące zapytanie użytkownika (nigdy nie przycinane)
3. Wyniki narzędzi (tool results) — przycinane proporcjonalnie
4. Najnowsze wiadomości z historii — przycinane od najstarszych
5. Snapshot / pamięć — przycinany w ostateczności
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TokenCounter:
    """
    Licznik tokenów z heurystycznym fallbackiem.

    Próbuje użyć tiktoken (OpenAI) jeśli dostępny,
    w przeciwnym razie stosuje heurystykę ~4 znaki = 1 token.
    """

    _encoder = None
    _encoder_loaded = False

    @classmethod
    def _load_encoder(cls) -> None:
        """Lazy-load tiktoken encoder."""
        if cls._encoder_loaded:
            return
        cls._encoder_loaded = True
        try:
            import tiktoken

            cls._encoder = tiktoken.get_encoding("cl100k_base")
            logger.info("TokenCounter: using tiktoken cl100k_base encoder")
        except ImportError:
            logger.info("TokenCounter: tiktoken not available, using heuristic")
        except Exception as e:
            logger.warning("TokenCounter: tiktoken load error: %s, using heuristic", e)

    @classmethod
    def count(cls, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        cls._load_encoder()

        if cls._encoder is not None:
            try:
                return len(cls._encoder.encode(text))
            except Exception:
                pass

        # Heuristic fallback: ~4 chars per token for English,
        # ~2-3 chars per token for Polish/Cyrillic
        # We use a conservative estimate
        return max(1, len(text) // 3)

    @classmethod
    def count_messages(cls, messages: list[dict[str, str]]) -> int:
        """
        Count tokens in a list of messages.

        Accounts for message overhead (role, formatting).
        Each message has ~4 tokens overhead.
        """
        total = 0
        for msg in messages:
            total += 4  # message overhead (role, separators)
            total += cls.count(msg.get("content", ""))
            total += cls.count(msg.get("role", ""))
        total += 2  # reply priming
        return total


# ---------------------------------------------------------------------------
# Priority system
# ---------------------------------------------------------------------------


class MessagePriority(IntEnum):
    """Priority levels for context messages (higher = more important)."""

    SNAPSHOT = 10
    TOOL_RESULT = 20
    HISTORY_OLD = 30
    HISTORY_RECENT = 40
    VERTEX_RESULT = 50
    RAG_RESULT = 55
    WEB_RESULT = 50
    SYSTEM_CONTEXT = 60
    USER_MEMORY = 65
    SYSTEM_PROMPT = 90
    CURRENT_QUERY = 100


@dataclass
class PrioritizedMessage:
    """Message with priority metadata for smart truncation."""

    message: dict[str, str]
    priority: MessagePriority
    token_count: int = 0
    truncatable: bool = True
    min_tokens: int = 50  # minimum tokens to keep if truncated
    source: str = ""  # identifier for logging

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = TokenCounter.count(self.message.get("content", ""))


# ---------------------------------------------------------------------------
# Model token limits
# ---------------------------------------------------------------------------

# Context window sizes for known models
MODEL_TOKEN_LIMITS: dict[str, int] = {
    # Gemini
    "gemini-2.0-flash": 1_000_000,
    "gemini-2.0-flash-thinking-exp": 32_000,
    "gemini-2.5-pro-preview-05-06": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    # OpenAI
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    # Claude
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    # DeepSeek
    "deepseek-chat": 64_000,
    "deepseek-coder": 64_000,
    # Groq (Llama)
    "llama-3.3-70b-versatile": 128_000,
    "llama-3.1-8b-instant": 128_000,
    "mixtral-8x7b-32768": 32_768,
    # Grok
    "grok-2": 128_000,
    "grok-2-mini": 128_000,
}

# Default limits per provider (fallback)
PROVIDER_DEFAULT_LIMITS: dict[str, int] = {
    "gemini": 1_000_000,
    "openai": 128_000,
    "claude": 200_000,
    "deepseek": 64_000,
    "groq": 128_000,
    "grok": 128_000,
    "openrouter": 64_000,
}


def get_model_token_limit(model: str, provider: str = "") -> int:
    """Get token limit for a model."""
    if model in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model]
    if provider in PROVIDER_DEFAULT_LIMITS:
        return PROVIDER_DEFAULT_LIMITS[provider]
    return 32_000  # conservative default


# ---------------------------------------------------------------------------
# Token Budget Manager
# ---------------------------------------------------------------------------


@dataclass
class BudgetReport:
    """Report from token budget management."""

    original_token_count: int
    final_token_count: int
    model_limit: int
    tokens_saved: int
    messages_truncated: int
    messages_removed: int
    budget_utilization: float = 0.0  # 0.0 - 1.0
    warnings: list[str] = field(default_factory=list)


class TokenBudgetManager:
    """
    Zarządza budżetem tokenów dla zapytań do LLM.

    Strategia smart truncation:
    1. Oblicz tokeny dla wszystkich wiadomości
    2. Jeśli mieści się w budżecie — zwróć bez zmian
    3. Jeśli przekracza — przycinaj od najniższego priorytetu:
       a) Skróć snapshot do podsumowania
       b) Usuń najstarsze wiadomości z historii
       c) Skróć wyniki narzędzi (zachowaj top-N)
       d) NIGDY nie usuwaj system prompt ani bieżącego zapytania
    """

    # Reserve tokens for model response
    RESPONSE_RESERVE_RATIO = 0.15  # 15% of context for response
    MIN_RESPONSE_RESERVE = 1024
    MAX_RESPONSE_RESERVE = 8192

    # Safety margin
    SAFETY_MARGIN_RATIO = 0.05  # 5% safety buffer

    def __init__(
        self,
        model: str = "",
        provider: str = "",
        max_context_tokens: int | None = None,
    ) -> None:
        """
        Initialize Token Budget Manager.

        Args:
            model: Model name for automatic limit detection
            provider: Provider name for fallback limit
            max_context_tokens: Override for max context tokens
        """
        self.model = model
        self.provider = provider

        # Determine token limit
        if max_context_tokens:
            self._model_limit = max_context_tokens
        else:
            self._model_limit = get_model_token_limit(model, provider)

        # Calculate effective budget
        response_reserve = min(
            max(
                int(self._model_limit * self.RESPONSE_RESERVE_RATIO),
                self.MIN_RESPONSE_RESERVE,
            ),
            self.MAX_RESPONSE_RESERVE,
        )
        safety_margin = int(self._model_limit * self.SAFETY_MARGIN_RATIO)
        self._effective_budget = self._model_limit - response_reserve - safety_margin

        logger.info(
            f"TokenBudgetManager initialized: model={model}, "
            f"limit={self._model_limit}, effective_budget={self._effective_budget}"
        )

    @property
    def model_limit(self) -> int:
        return self._model_limit

    @property
    def effective_budget(self) -> int:
        return self._effective_budget

    def fits_in_budget(self, messages: list[dict[str, str]]) -> bool:
        """Check if messages fit within the token budget."""
        return TokenCounter.count_messages(messages) <= self._effective_budget

    def apply_budget(
        self,
        prioritized_messages: list[PrioritizedMessage],
    ) -> tuple[list[dict[str, str]], BudgetReport]:
        """
        Apply token budget to prioritized messages.

        Performs smart truncation to fit within budget.

        Args:
            prioritized_messages: Messages with priority metadata

        Returns:
            Tuple of (final messages list, budget report)
        """
        # Calculate initial token count
        total_tokens = sum(pm.token_count for pm in prioritized_messages)
        original_tokens = total_tokens

        report = BudgetReport(
            original_token_count=original_tokens,
            final_token_count=0,
            model_limit=self._model_limit,
            tokens_saved=0,
            messages_truncated=0,
            messages_removed=0,
        )

        # If fits — return as-is
        if total_tokens <= self._effective_budget:
            final_messages = [pm.message for pm in prioritized_messages]
            report.final_token_count = total_tokens
            report.budget_utilization = (
                total_tokens / self._effective_budget if self._effective_budget > 0 else 0
            )
            logger.info(
                f"Context fits in budget: {total_tokens}/{self._effective_budget} tokens "
                f"({report.budget_utilization:.1%})"
            )
            return final_messages, report

        # Need truncation — sort by priority (ascending = lowest first to truncate)
        logger.warning(
            f"Context exceeds budget: {total_tokens}/{self._effective_budget} tokens. "
            f"Starting smart truncation..."
        )

        # Work on a copy sorted by priority (lowest first for truncation)
        truncation_order = sorted(
            [pm for pm in prioritized_messages if pm.truncatable],
            key=lambda pm: pm.priority,
        )
        [pm for pm in prioritized_messages if not pm.truncatable]

        # Phase 1: Remove lowest-priority messages entirely
        tokens_to_save = total_tokens - self._effective_budget
        removed_indices: set[int] = set()

        for pm in truncation_order:
            if tokens_to_save <= 0:
                break
            idx = prioritized_messages.index(pm)
            removed_indices.add(idx)
            tokens_to_save -= pm.token_count
            total_tokens -= pm.token_count
            report.messages_removed += 1
            logger.debug(
                f"Removed message (priority={pm.priority}, source={pm.source}), "
                f"saved {pm.token_count} tokens"
            )

        # Phase 2: If still over budget, truncate remaining truncatable messages
        if total_tokens > self._effective_budget:
            remaining = [
                (i, pm)
                for i, pm in enumerate(prioritized_messages)
                if i not in removed_indices and pm.truncatable
            ]
            remaining.sort(key=lambda x: x[1].priority)

            for _, pm in remaining:
                if total_tokens <= self._effective_budget:
                    break

                content = pm.message.get("content", "")
                current_tokens = pm.token_count
                target_tokens = max(pm.min_tokens, current_tokens // 2)

                # Truncate content
                truncated = self._smart_truncate_text(content, target_tokens)
                tokens_saved = current_tokens - TokenCounter.count(truncated)

                pm.message["content"] = truncated
                pm.token_count -= tokens_saved
                total_tokens -= tokens_saved
                report.messages_truncated += 1

                logger.debug(
                    f"Truncated message (priority={pm.priority}, source={pm.source}), "
                    f"saved {tokens_saved} tokens"
                )

        # Build final message list preserving original order
        final_messages = []
        for i, pm in enumerate(prioritized_messages):
            if i not in removed_indices:
                final_messages.append(pm.message)

        report.final_token_count = total_tokens
        report.tokens_saved = original_tokens - total_tokens
        report.budget_utilization = (
            total_tokens / self._effective_budget if self._effective_budget > 0 else 0
        )

        if total_tokens > self._effective_budget:
            report.warnings.append(
                f"Kontekst nadal przekracza budżet po truncation: "
                f"{total_tokens}/{self._effective_budget}"
            )

        logger.info(
            f"Smart truncation complete: {original_tokens} → {total_tokens} tokens "
            f"(saved {report.tokens_saved}, removed {report.messages_removed}, "
            f"truncated {report.messages_truncated})"
        )

        return final_messages, report

    def _smart_truncate_text(self, text: str, target_tokens: int) -> str:
        """
        Intelligently truncate text to target token count.

        Strategy:
        - Keep first paragraph (usually most important)
        - Keep last paragraph (usually conclusion)
        - Truncate middle content
        """
        if not text:
            return text

        current_tokens = TokenCounter.count(text)
        if current_tokens <= target_tokens:
            return text

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if len(paragraphs) <= 2:
            # Simple truncation — keep beginning
            chars_to_keep = max(100, target_tokens * 3)  # rough estimate
            return text[:chars_to_keep] + "\n\n[...treść skrócona...]"

        # Keep first and last paragraph, truncate middle
        first = paragraphs[0]
        last = paragraphs[-1]
        first_tokens = TokenCounter.count(first)
        last_tokens = TokenCounter.count(last)

        remaining_budget = target_tokens - first_tokens - last_tokens - 10  # overhead

        if remaining_budget <= 0:
            # Even first+last don't fit — just keep first
            chars_to_keep = max(100, target_tokens * 3)
            return text[:chars_to_keep] + "\n\n[...treść skrócona...]"

        # Fill middle with as many paragraphs as fit
        middle_parts = []
        middle_tokens = 0
        for p in paragraphs[1:-1]:
            p_tokens = TokenCounter.count(p)
            if middle_tokens + p_tokens <= remaining_budget:
                middle_parts.append(p)
                middle_tokens += p_tokens
            else:
                break

        if middle_parts:
            truncation_note = (
                f"\n\n[...pominięto {len(paragraphs) - 2 - len(middle_parts)} akapitów...]"
            )
            return first + "\n\n" + "\n\n".join(middle_parts) + truncation_note + "\n\n" + last
        else:
            return first + "\n\n[...treść skrócona...]\n\n" + last

    def estimate_cost(
        self,
        messages: list[dict[str, str]],
        provider: str = "",
        model: str = "",
    ) -> dict[str, float]:
        """
        Estimate cost for sending these messages.

        Returns:
            Dict with input_tokens, estimated_output_tokens, estimated_cost_usd
        """
        from app.services.model_router import ModelRouter

        input_tokens = TokenCounter.count_messages(messages)
        estimated_output = min(input_tokens // 2, 2048)  # rough estimate

        router = ModelRouter()
        cost_estimate = router.estimate_cost(
            profile="smart",  # default
            provider=provider or self.provider,
            input_tokens=input_tokens,
            output_tokens=estimated_output,
        )

        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output,
            "estimated_cost_usd": cost_estimate.estimated_cost_usd,
        }
