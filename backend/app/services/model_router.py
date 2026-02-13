"""
Model Router — inteligentny routing modeli i narzędzi.

Odpowiada za:
- Zaawansowaną klasyfikację trudności zapytań (multi-signal)
- Automatyczny dobór profilu (eco/smart/deep)
- Inteligentny wybór narzędzi na podstawie analizy zapytania
- Szacowanie kosztów
- Ocenę trafności narzędzi (tool relevance scoring)

Strategia klasyfikacji:
1. Analiza słów kluczowych (PL + EN)
2. Analiza struktury zapytania (długość, złożoność syntaktyczna)
3. Detekcja intencji (pytanie o fakty, analiza, kod, obliczenia, czas)
4. Scoring wielosygnałowy z wagami
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


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


class QueryIntent(str, Enum):
    """Detected intent of the user query."""
    FACTUAL = "factual"           # Simple fact lookup
    ANALYTICAL = "analytical"     # Analysis, comparison, reasoning
    CREATIVE = "creative"         # Creative writing, brainstorming
    CODE = "code"                 # Programming, debugging
    MATH = "math"                 # Calculations, math problems
    SEARCH = "search"             # Web/knowledge search needed
    DOCUMENT = "document"         # Document-related query
    MEMORY = "memory"             # User preference/memory related
    TEMPORAL = "temporal"         # Time/date related
    CONVERSATIONAL = "conversational"  # Casual chat, greetings


@dataclass
class CostEstimate:
    """Cost estimation for a profile."""
    profile: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    provider: str


@dataclass
class ToolRecommendation:
    """Recommendation for which tools to use."""
    tool_name: str
    relevance_score: float  # 0.0 - 1.0
    reason: str


@dataclass
class QueryAnalysis:
    """Complete analysis of a user query."""
    difficulty: DifficultyLevel
    intents: list[QueryIntent]
    recommended_tools: list[ToolRecommendation]
    confidence: float  # 0.0 - 1.0
    signals: dict[str, Any]  # debug info about classification signals


class ModelRouter:
    """Router for model selection based on difficulty and profile."""

    # =========================================================================
    # Keyword dictionaries (Polish + English)
    # =========================================================================

    # Hard difficulty indicators
    HARD_KEYWORDS_PL = [
        "wyjaśnij szczegółowo", "przeanalizuj", "porównaj", "zaprojektuj",
        "zoptymalizuj", "debuguj", "refaktoryzuj", "architektura",
        "algorytm", "złożoność", "zaimplementuj", "strategia",
        "oceń", "zaproponuj rozwiązanie", "rozwiąż problem",
        "wieloetapowy", "krok po kroku",
    ]

    HARD_KEYWORDS_EN = [
        "explain in detail", "analyze", "compare", "design",
        "optimize", "debug", "refactor", "architecture",
        "algorithm", "complexity", "implement", "strategy",
        "evaluate", "propose solution", "solve problem",
        "multi-step", "step by step",
    ]

    # Medium difficulty indicators
    MEDIUM_KEYWORDS_PL = [
        "jak", "dlaczego", "co to jest", "różnica", "przykład",
        "pokaż", "napisz", "stwórz", "opisz", "wymień",
        "podaj", "wytłumacz", "pomóż",
    ]

    MEDIUM_KEYWORDS_EN = [
        "how", "why", "what is", "difference", "example",
        "show", "write", "create", "describe", "list",
        "give me", "explain", "help",
    ]

    # Intent detection patterns
    CODE_PATTERNS = [
        r"\b(kod|code|python|javascript|java|c\+\+|rust|go|sql|html|css)\b",
        r"\b(funkcj[aęi]|function|class|metod[aęy]|method|api|endpoint)\b",
        r"\b(bug|błąd|error|exception|debug|test|deploy)\b",
        r"\b(git|github|docker|kubernetes|ci/cd)\b",
        r"```",  # code block
    ]

    MATH_PATTERNS = [
        r"\b(oblicz|policz|calculate|compute|ile|how much|how many)\b",
        r"\b(procent|percent|%|suma|sum|średnia|average|mediana|median)\b",
        r"\b(równanie|equation|wzór|formula|integral|pochodna|derivative)\b",
        r"[\d+\-*/^()]{3,}",  # math expressions
    ]

    SEARCH_PATTERNS = [
        r"\b(znajdź|find|szukaj|search|wyszukaj|look up)\b",
        r"\b(aktualn[eya]|current|najnowsz[eya]|latest|recent|dzisiaj|today)\b",
        r"\b(cena|price|pogoda|weather|kurs|rate|news|wiadomości)\b",
        r"\b(strona|website|link|url|artykuł|article)\b",
    ]

    DOCUMENT_PATTERNS = [
        r"\b(dokument|document|plik|file|pdf|docx|notatk[ai]|note)\b",
        r"\b(przesłan[eya]|uploaded|moj[eai]|my|moje pliki|my files)\b",
        r"\b(treść|content|fragment|excerpt|cytat|quote)\b",
    ]

    MEMORY_PATTERNS = [
        r"\b(zapamiętaj|remember|zapamięt|pamiętasz|do you remember)\b",
        r"\b(moje? imi[eę]|my name|preferenc[jei]|preference)\b",
        r"\b(ulubion[eya]|favorite|favourite)\b",
        r"\b(ustaw|set|zmień|change|aktualizuj|update)\s+(moj|my)\b",
    ]

    TEMPORAL_PATTERNS = [
        r"\b(czas|time|data|date|godzina|hour|dzisiaj|today|teraz|now)\b",
        r"\b(kiedy|when|który rok|what year|jaki dzień|what day)\b",
    ]

    CONVERSATIONAL_PATTERNS = [
        r"^(cześć|hej|siema|hello|hi|hey|yo|witaj|dzień dobry|good morning)\b",
        r"^(dzięki|thanks|thank you|dziękuję|ok|okay|super|great|fajnie)\b",
        r"^(tak|nie|yes|no|pewnie|sure|oczywiście|of course)\b",
        r"\b(jak się masz|how are you|co słychać|what's up)\b",
    ]

    # Token-based smart credits calculation
    SMART_CREDIT_TIERS = [
        (500, 1),    # ≤500 tokens = 1 credit
        (2000, 2),   # ≤2000 tokens = 2 credits
        (float("inf"), 4),  # >2000 tokens = 4 credits
    ]

    # Cost estimates (USD per 1M tokens)
    COST_ESTIMATES = {
        "gemini": {"input": 0.075, "output": 0.30},
        "deepseek": {"input": 0.14, "output": 0.28},
        "groq": {"input": 0.0, "output": 0.0},
        "openrouter": {"input": 0.0, "output": 0.0},
        "grok": {"input": 5.0, "output": 15.0},
        "openai": {"input": 2.5, "output": 10.0},
        "claude": {"input": 3.0, "output": 15.0},
    }

    # =========================================================================
    # Main classification methods
    # =========================================================================

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Perform comprehensive query analysis.

        Multi-signal classification:
        1. Keyword matching (PL + EN)
        2. Pattern-based intent detection
        3. Structural analysis (length, complexity)
        4. Tool relevance scoring

        Args:
            query: User query text

        Returns:
            QueryAnalysis with difficulty, intents, and tool recommendations
        """
        query_lower = query.lower().strip()
        signals: dict[str, Any] = {}

        # --- Signal 1: Keyword difficulty scoring ---
        hard_score = self._keyword_score(query_lower, self.HARD_KEYWORDS_PL + self.HARD_KEYWORDS_EN)
        medium_score = self._keyword_score(query_lower, self.MEDIUM_KEYWORDS_PL + self.MEDIUM_KEYWORDS_EN)
        signals["hard_keyword_score"] = hard_score
        signals["medium_keyword_score"] = medium_score

        # --- Signal 2: Structural complexity ---
        word_count = len(query.split())
        sentence_count = max(1, len(re.split(r'[.!?]+', query)))
        has_code_block = "```" in query
        has_list = bool(re.search(r'^\s*[-*\d]+[.)]\s', query, re.MULTILINE))

        structural_complexity = 0.0
        if word_count > 100:
            structural_complexity = 1.0
        elif word_count > 50:
            structural_complexity = 0.7
        elif word_count > 20:
            structural_complexity = 0.4
        elif word_count > 10:
            structural_complexity = 0.2

        if has_code_block:
            structural_complexity = min(1.0, structural_complexity + 0.3)
        if has_list:
            structural_complexity = min(1.0, structural_complexity + 0.1)
        if sentence_count > 3:
            structural_complexity = min(1.0, structural_complexity + 0.2)

        signals["word_count"] = word_count
        signals["structural_complexity"] = structural_complexity

        # --- Signal 3: Intent detection ---
        intents = self._detect_intents(query_lower, query)
        signals["intents"] = [i.value for i in intents]

        # --- Signal 4: Combined difficulty score ---
        difficulty_score = (
            hard_score * 0.4 +
            structural_complexity * 0.3 +
            (0.3 if any(i in (QueryIntent.ANALYTICAL, QueryIntent.CODE) for i in intents) else 0.0) +
            medium_score * 0.15
        )

        # Reduce difficulty for conversational/simple queries
        if QueryIntent.CONVERSATIONAL in intents and len(intents) == 1:
            difficulty_score = max(0.0, difficulty_score - 0.5)

        signals["difficulty_score"] = difficulty_score

        # --- Classify difficulty ---
        if difficulty_score >= 0.6:
            difficulty = DifficultyLevel.HARD
        elif difficulty_score >= 0.25:
            difficulty = DifficultyLevel.MEDIUM
        else:
            difficulty = DifficultyLevel.EASY

        # --- Signal 5: Tool recommendations ---
        recommended_tools = self._recommend_tools(query_lower, query, intents)
        signals["recommended_tool_count"] = len(recommended_tools)

        # Confidence based on signal agreement
        confidence = min(1.0, 0.5 + abs(difficulty_score - 0.35) * 1.5)

        return QueryAnalysis(
            difficulty=difficulty,
            intents=intents,
            recommended_tools=recommended_tools,
            confidence=confidence,
            signals=signals,
        )

    def classify_difficulty(self, query: str) -> DifficultyLevel:
        """
        Classify query difficulty (backward-compatible interface).

        Uses the full analyze_query pipeline internally.

        Args:
            query: User query text

        Returns:
            DifficultyLevel (easy, medium, hard)
        """
        analysis = self.analyze_query(query)
        return analysis.difficulty

    def get_recommended_tools(self, query: str) -> list[ToolRecommendation]:
        """
        Get tool recommendations for a query.

        Args:
            query: User query text

        Returns:
            List of ToolRecommendation sorted by relevance
        """
        analysis = self.analyze_query(query)
        return analysis.recommended_tools

    # =========================================================================
    # Profile selection
    # =========================================================================

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
                if user_role in ("FULL_ACCESS", "ADMIN"):
                    return Profile.DEEP
                else:
                    return Profile.SMART
            elif mode_lower == "smart":
                return Profile.SMART
            elif mode_lower == "eco":
                return Profile.ECO

        # Automatic selection based on difficulty
        if difficulty == DifficultyLevel.HARD:
            if user_role == "DEMO":
                return Profile.SMART
            else:
                return Profile.DEEP
        elif difficulty == DifficultyLevel.MEDIUM:
            return Profile.SMART
        else:
            return Profile.ECO

    # =========================================================================
    # Cost estimation
    # =========================================================================

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
            profile=profile.value if isinstance(profile, Profile) else profile,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=total_cost,
            provider=provider,
        )

    def calculate_smart_credits(self, total_tokens: int) -> int:
        """
        Calculate smart credits based on token count.

        Args:
            total_tokens: Total tokens (input + output)

        Returns:
            Smart credits consumed
        """
        for threshold, credits in self.SMART_CREDIT_TIERS:
            if total_tokens <= threshold:
                return credits
        return 4

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

    # =========================================================================
    # Private methods
    # =========================================================================

    def _keyword_score(self, query_lower: str, keywords: list[str]) -> float:
        """Calculate keyword match score (0.0 - 1.0)."""
        matches = sum(1 for kw in keywords if kw in query_lower)
        if matches == 0:
            return 0.0
        # Logarithmic scaling: 1 match = 0.3, 2 = 0.5, 3+ = 0.7+
        import math
        return min(1.0, 0.3 + math.log(1 + matches) * 0.3)

    def _detect_intents(self, query_lower: str, query_original: str) -> list[QueryIntent]:
        """Detect query intents using pattern matching."""
        intents: list[QueryIntent] = []

        pattern_map = [
            (self.CODE_PATTERNS, QueryIntent.CODE),
            (self.MATH_PATTERNS, QueryIntent.MATH),
            (self.SEARCH_PATTERNS, QueryIntent.SEARCH),
            (self.DOCUMENT_PATTERNS, QueryIntent.DOCUMENT),
            (self.MEMORY_PATTERNS, QueryIntent.MEMORY),
            (self.TEMPORAL_PATTERNS, QueryIntent.TEMPORAL),
            (self.CONVERSATIONAL_PATTERNS, QueryIntent.CONVERSATIONAL),
        ]

        for patterns, intent in pattern_map:
            for pattern in patterns:
                try:
                    if re.search(pattern, query_lower, re.IGNORECASE):
                        if intent not in intents:
                            intents.append(intent)
                        break
                except re.error:
                    continue

        # Default intent if none detected
        if not intents:
            # Check if it's a question
            if query_lower.endswith("?") or any(
                query_lower.startswith(w) for w in ["co ", "kto ", "gdzie ", "kiedy ", "jak ", "dlaczego ",
                                                     "what ", "who ", "where ", "when ", "how ", "why "]
            ):
                intents.append(QueryIntent.FACTUAL)
            else:
                intents.append(QueryIntent.CONVERSATIONAL)

        return intents

    def _recommend_tools(
        self,
        query_lower: str,
        query_original: str,
        intents: list[QueryIntent],
    ) -> list[ToolRecommendation]:
        """
        Recommend tools based on detected intents and query content.

        Returns tools sorted by relevance score (descending).
        """
        recommendations: list[ToolRecommendation] = []

        # Web search — for search intent, current info, factual queries
        if QueryIntent.SEARCH in intents:
            recommendations.append(ToolRecommendation(
                tool_name="web_search",
                relevance_score=0.9,
                reason="Zapytanie wymaga aktualnych informacji z internetu.",
            ))
        elif QueryIntent.FACTUAL in intents:
            # Lower relevance — might be answerable from knowledge
            recommendations.append(ToolRecommendation(
                tool_name="web_search",
                relevance_score=0.4,
                reason="Zapytanie faktograficzne — wyszukiwanie może pomóc.",
            ))

        # Vertex search — for knowledge base queries
        if QueryIntent.SEARCH in intents or QueryIntent.FACTUAL in intents:
            recommendations.append(ToolRecommendation(
                tool_name="vertex_search",
                relevance_score=0.5,
                reason="Zapytanie może dotyczyć bazy wiedzy.",
            ))

        # RAG search — for document-related queries
        if QueryIntent.DOCUMENT in intents:
            recommendations.append(ToolRecommendation(
                tool_name="rag_search",
                relevance_score=0.85,
                reason="Zapytanie dotyczy dokumentów użytkownika.",
            ))

        # Calculator — for math queries
        if QueryIntent.MATH in intents:
            recommendations.append(ToolRecommendation(
                tool_name="calculate",
                relevance_score=0.8,
                reason="Zapytanie wymaga obliczeń matematycznych.",
            ))

        # Memory — for memory-related queries
        if QueryIntent.MEMORY in intents:
            recommendations.append(ToolRecommendation(
                tool_name="memory_read",
                relevance_score=0.7,
                reason="Zapytanie dotyczy zapamiętanych preferencji.",
            ))
            recommendations.append(ToolRecommendation(
                tool_name="memory_write",
                relevance_score=0.6,
                reason="Użytkownik może chcieć zapisać informację.",
            ))

        # DateTime — for temporal queries
        if QueryIntent.TEMPORAL in intents:
            recommendations.append(ToolRecommendation(
                tool_name="get_datetime",
                relevance_score=0.8,
                reason="Zapytanie dotyczy aktualnej daty/czasu.",
            ))

        # Sort by relevance (descending)
        recommendations.sort(key=lambda r: r.relevance_score, reverse=True)

        return recommendations
