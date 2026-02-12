"""
Model router for difficulty classification and profile selection.
"""

from dataclasses import dataclass
from enum import Enum


class DifficultyLevel(str, Enum):
    """Difficulty levels for query classification."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Profile(str, Enum):
    """AI profile types."""

    ECO = "eco"
    SMART = "smart"
    DEEP = "deep"


@dataclass
class CostEstimate:
    """Cost estimation for a profile."""

    profile: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    provider: str


class ModelRouter:
    """Router for model selection based on difficulty and profile."""

    # Keywords for difficulty classification (Polish + English)
    HARD_KEYWORDS_PL = [
        "wyjaśnij szczegółowo",
        "przeanalizuj",
        "porównaj",
        "zaprojektuj",
        "zoptymalizuj",
        "debuguj",
        "refaktoryzuj",
        "architektura",
        "algorytm",
        "złożoność",
    ]

    HARD_KEYWORDS_EN = [
        "explain in detail",
        "analyze",
        "compare",
        "design",
        "optimize",
        "debug",
        "refactor",
        "architecture",
        "algorithm",
        "complexity",
    ]

    MEDIUM_KEYWORDS_PL = [
        "jak",
        "dlaczego",
        "co to jest",
        "różnica",
        "przykład",
        "pokaż",
        "napisz",
        "stwórz",
    ]

    MEDIUM_KEYWORDS_EN = [
        "how",
        "why",
        "what is",
        "difference",
        "example",
        "show",
        "write",
        "create",
    ]

    # Token-based smart credits calculation
    SMART_CREDIT_TIERS = [
        (500, 1),  # ≤500 tokens = 1 credit
        (2000, 2),  # ≤2000 tokens = 2 credits
        (float("inf"), 4),  # >2000 tokens = 4 credits
    ]

    # Cost estimates (USD per 1M tokens)
    COST_ESTIMATES = {
        "gemini": {"input": 0.075, "output": 0.30},  # Flash
        "deepseek": {"input": 0.14, "output": 0.28},  # Chat
        "groq": {"input": 0.0, "output": 0.0},  # Free tier
        "openrouter": {"input": 0.0, "output": 0.0},  # Free tier
        "grok": {"input": 5.0, "output": 15.0},
        "openai": {"input": 2.5, "output": 10.0},  # GPT-4o
        "claude": {"input": 3.0, "output": 15.0},  # Sonnet
    }

    def classify_difficulty(self, query: str) -> DifficultyLevel:
        """
        Classify query difficulty using heuristics.

        Args:
            query: User query text

        Returns:
            DifficultyLevel (easy, medium, hard)
        """
        query_lower = query.lower()

        # Check for hard keywords
        hard_keywords = self.HARD_KEYWORDS_PL + self.HARD_KEYWORDS_EN
        if any(keyword in query_lower for keyword in hard_keywords):
            return DifficultyLevel.HARD

        # Check for medium keywords
        medium_keywords = self.MEDIUM_KEYWORDS_PL + self.MEDIUM_KEYWORDS_EN
        if any(keyword in query_lower for keyword in medium_keywords):
            return DifficultyLevel.MEDIUM

        # Check query length
        word_count = len(query.split())
        if word_count > 50:
            return DifficultyLevel.HARD
        elif word_count > 20:
            return DifficultyLevel.MEDIUM

        # Default to easy
        return DifficultyLevel.EASY

    def select_profile(
        self,
        difficulty: DifficultyLevel,
        user_mode: str | None = None,
        user_role: str = "DEMO",
    ) -> Profile:
        """
        Select AI profile based on difficulty and user preferences.

        Args:
            difficulty: Classified difficulty level
            user_mode: User's preferred mode override
            user_role: User's role (DEMO, FULL_ACCESS, ADMIN)

        Returns:
            Profile (eco, smart, deep)
        """
        # User override takes precedence
        if user_mode:
            mode_lower = user_mode.lower()
            if mode_lower == "deep":
                # DEEP requires FULL_ACCESS or higher
                if user_role in ("FULL_ACCESS", "ADMIN"):
                    return Profile.DEEP
                else:
                    # Fallback to SMART for DEMO users
                    return Profile.SMART
            elif mode_lower == "smart":
                return Profile.SMART
            elif mode_lower == "eco":
                return Profile.ECO

        # Automatic selection based on difficulty
        if difficulty == DifficultyLevel.HARD:
            # DEMO users get SMART, others get DEEP
            if user_role == "DEMO":
                return Profile.SMART
            else:
                return Profile.DEEP
        elif difficulty == DifficultyLevel.MEDIUM:
            return Profile.SMART
        else:
            return Profile.ECO

    def estimate_cost(
        self,
        profile: Profile,
        provider: str,
        input_tokens: int,
        output_tokens: int = 500,
    ) -> CostEstimate:
        """
        Estimate cost for a request.

        Args:
            profile: AI profile
            provider: Provider name
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens

        Returns:
            CostEstimate with breakdown
        """
        costs = self.COST_ESTIMATES.get(provider, {"input": 0.0, "output": 0.0})

        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        total_cost = input_cost + output_cost

        return CostEstimate(
            profile=profile.value,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=total_cost,
            provider=provider,
        )

    def calculate_smart_credits(self, total_tokens: int) -> int:
        """
        Calculate smart credits based on token count.

        Token tiers:
        - ≤500 tokens = 1 credit
        - ≤2000 tokens = 2 credits
        - >2000 tokens = 4 credits

        Args:
            total_tokens: Total tokens (input + output)

        Returns:
            Smart credits consumed
        """
        for threshold, credits in self.SMART_CREDIT_TIERS:
            if total_tokens <= threshold:
                return credits

        return 4  # Fallback

    def needs_confirmation(self, profile: Profile, user_role: str) -> bool:
        """
        Check if profile requires user confirmation.

        DEEP mode requires confirmation for FULL_ACCESS users.

        Args:
            profile: Selected profile
            user_role: User's role

        Returns:
            True if confirmation needed
        """
        if profile == Profile.DEEP and user_role == "FULL_ACCESS":
            return True

        return False
