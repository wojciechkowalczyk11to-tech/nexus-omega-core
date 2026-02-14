"""
Tests for Phase 2: Agent Architecture components.

Covers:
- ToolRegistry: registration, schema generation, execution
- TokenBudgetManager: counting, budgeting, smart truncation
- ModelRouter: multi-signal classification, intent detection, tool recommendations
- Orchestrator: ReAct loop data models
"""

import asyncio
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Tool Registry Tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def _make_registry(self):
        from app.tools.tool_registry import (
            ParameterType,
            ToolDefinition,
            ToolParameter,
            ToolRegistry,
            ToolResult,
        )

        registry = ToolRegistry()

        async def dummy_handler(query: str, max_results: int = 5) -> ToolResult:
            return ToolResult(success=True, data=f"Results for: {query}")

        registry.register(
            ToolDefinition(
                name="test_search",
                description="Test search tool",
                parameters=[
                    ToolParameter(
                        name="query",
                        type=ParameterType.STRING,
                        description="Search query",
                    ),
                    ToolParameter(
                        name="max_results",
                        type=ParameterType.INTEGER,
                        description="Max results",
                        required=False,
                        default=5,
                    ),
                ],
                handler=dummy_handler,
                category="search",
            )
        )
        return registry

    def test_register_and_list(self):
        registry = self._make_registry()
        assert "test_search" in registry.list_tool_names()
        assert len(registry.list_tools()) == 1

    def test_get_tool(self):
        registry = self._make_registry()
        tool = registry.get("test_search")
        assert tool is not None
        assert tool.name == "test_search"
        assert tool.category == "search"

    def test_get_nonexistent_tool(self):
        registry = self._make_registry()
        assert registry.get("nonexistent") is None

    def test_unregister(self):
        registry = self._make_registry()
        assert registry.unregister("test_search") is True
        assert registry.get("test_search") is None
        assert registry.unregister("test_search") is False

    def test_openai_schema(self):
        registry = self._make_registry()
        schemas = registry.get_openai_tools()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_search"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]

    def test_gemini_schema(self):
        registry = self._make_registry()
        schemas = registry.get_gemini_tools()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["name"] == "test_search"
        assert schema["parameters"]["type"] == "OBJECT"
        # Gemini uses uppercase types
        assert schema["parameters"]["properties"]["query"]["type"] == "STRING"

    def test_claude_schema(self):
        registry = self._make_registry()
        schemas = registry.get_claude_tools()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["name"] == "test_search"
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"

    def test_provider_schema_routing(self):
        registry = self._make_registry()
        # OpenAI-compatible providers
        for provider in ["openai", "deepseek", "groq", "grok", "openrouter"]:
            schemas = registry.get_tools_for_provider(provider)
            assert schemas[0]["type"] == "function"

        # Claude
        schemas = registry.get_tools_for_provider("claude")
        assert "input_schema" in schemas[0]

        # Gemini
        schemas = registry.get_tools_for_provider("gemini")
        assert schemas[0]["parameters"]["type"] == "OBJECT"

    def test_tool_descriptions(self):
        registry = self._make_registry()
        desc = registry.get_tool_descriptions()
        assert "test_search" in desc
        assert "query" in desc

    @pytest.mark.asyncio
    async def test_execute_success(self):
        registry = self._make_registry()
        result = await registry.execute("test_search", {"query": "hello"})
        assert result.success is True
        assert "hello" in result.data

    @pytest.mark.asyncio
    async def test_execute_missing_tool(self):
        registry = self._make_registry()
        result = await registry.execute("nonexistent", {"query": "hello"})
        assert result.success is False
        assert "nie jest zarejestrowane" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_required_param(self):
        registry = self._make_registry()
        result = await registry.execute("test_search", {})
        assert result.success is False
        assert "Brakujący wymagany parametr" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_default_param(self):
        registry = self._make_registry()
        result = await registry.execute("test_search", {"query": "test"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        from app.tools.tool_registry import (
            ParameterType,
            ToolDefinition,
            ToolParameter,
            ToolRegistry,
            ToolResult,
        )

        registry = ToolRegistry()

        async def slow_handler(query: str) -> ToolResult:
            await asyncio.sleep(10)
            return ToolResult(success=True, data="done")

        registry.register(
            ToolDefinition(
                name="slow_tool",
                description="Slow tool",
                parameters=[
                    ToolParameter(name="query", type=ParameterType.STRING, description="q"),
                ],
                handler=slow_handler,
                timeout_seconds=0.1,
            )
        )

        result = await registry.execute("slow_tool", {"query": "test"})
        assert result.success is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_execute_error_handling(self):
        from app.tools.tool_registry import (
            ParameterType,
            ToolDefinition,
            ToolParameter,
            ToolRegistry,
            ToolResult,
        )

        registry = ToolRegistry()

        async def error_handler(query: str) -> ToolResult:
            raise ValueError("Test error")

        registry.register(
            ToolDefinition(
                name="error_tool",
                description="Error tool",
                parameters=[
                    ToolParameter(name="query", type=ParameterType.STRING, description="q"),
                ],
                handler=error_handler,
            )
        )

        result = await registry.execute("error_tool", {"query": "test"})
        assert result.success is False
        assert "ValueError" in result.error

    def test_disabled_tool(self):
        from app.tools.tool_registry import (
            ParameterType,
            ToolDefinition,
            ToolParameter,
            ToolRegistry,
            ToolResult,
        )

        registry = ToolRegistry()

        async def handler(query: str) -> ToolResult:
            return ToolResult(success=True, data="ok")

        registry.register(
            ToolDefinition(
                name="disabled",
                description="Disabled tool",
                parameters=[
                    ToolParameter(name="query", type=ParameterType.STRING, description="q"),
                ],
                handler=handler,
                enabled=False,
            )
        )

        assert len(registry.list_tools(enabled_only=True)) == 0
        assert len(registry.list_tools(enabled_only=False)) == 1

    def test_execution_stats(self):
        registry = self._make_registry()
        stats = registry.get_stats()
        assert "test_search" in stats
        assert stats["test_search"]["calls"] == 0


# ---------------------------------------------------------------------------
# Token Budget Manager Tests
# ---------------------------------------------------------------------------


class TestTokenCounter:
    """Tests for TokenCounter."""

    def test_count_empty(self):
        from app.services.token_budget_manager import TokenCounter

        assert TokenCounter.count("") == 0

    def test_count_simple(self):
        from app.services.token_budget_manager import TokenCounter

        count = TokenCounter.count("Hello, world!")
        assert count > 0

    def test_count_polish(self):
        from app.services.token_budget_manager import TokenCounter

        count = TokenCounter.count("Cześć, jak się masz? To jest test tokenizacji.")
        assert count > 0

    def test_count_messages(self):
        from app.services.token_budget_manager import TokenCounter

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        count = TokenCounter.count_messages(messages)
        assert count > 0
        # Should be more than just text tokens (includes overhead)
        text_tokens = TokenCounter.count("You are a helpful assistant.") + TokenCounter.count(
            "Hello!"
        )
        assert count > text_tokens


class TestTokenBudgetManager:
    """Tests for TokenBudgetManager."""

    def test_initialization(self):
        from app.services.token_budget_manager import TokenBudgetManager

        manager = TokenBudgetManager(model="gpt-4o", provider="openai")
        assert manager.model_limit == 128_000
        assert manager.effective_budget > 0
        assert manager.effective_budget < manager.model_limit

    def test_initialization_custom_limit(self):
        from app.services.token_budget_manager import TokenBudgetManager

        manager = TokenBudgetManager(max_context_tokens=10_000)
        assert manager.model_limit == 10_000

    def test_fits_in_budget(self):
        from app.services.token_budget_manager import TokenBudgetManager

        manager = TokenBudgetManager(max_context_tokens=100_000)
        messages = [{"role": "user", "content": "Hello!"}]
        assert manager.fits_in_budget(messages) is True

    def test_apply_budget_no_truncation(self):
        from app.services.token_budget_manager import (
            MessagePriority,
            PrioritizedMessage,
            TokenBudgetManager,
        )

        manager = TokenBudgetManager(max_context_tokens=100_000)

        prioritized = [
            PrioritizedMessage(
                message={"role": "system", "content": "System prompt"},
                priority=MessagePriority.SYSTEM_PROMPT,
                truncatable=False,
                source="system",
            ),
            PrioritizedMessage(
                message={"role": "user", "content": "Hello!"},
                priority=MessagePriority.CURRENT_QUERY,
                truncatable=False,
                source="query",
            ),
        ]

        messages, report = manager.apply_budget(prioritized)
        assert len(messages) == 2
        assert report.tokens_saved == 0
        assert report.messages_removed == 0

    def test_apply_budget_with_truncation(self):
        from app.services.token_budget_manager import (
            MessagePriority,
            PrioritizedMessage,
            TokenBudgetManager,
        )

        # Very small budget to force truncation
        manager = TokenBudgetManager(max_context_tokens=100)

        long_text = "To jest bardzo długi tekst. " * 100

        prioritized = [
            PrioritizedMessage(
                message={"role": "system", "content": "System prompt"},
                priority=MessagePriority.SYSTEM_PROMPT,
                truncatable=False,
                source="system",
            ),
            PrioritizedMessage(
                message={"role": "system", "content": long_text},
                priority=MessagePriority.SNAPSHOT,
                truncatable=True,
                source="snapshot",
            ),
            PrioritizedMessage(
                message={"role": "user", "content": "Hello!"},
                priority=MessagePriority.CURRENT_QUERY,
                truncatable=False,
                source="query",
            ),
        ]

        messages, report = manager.apply_budget(prioritized)
        assert report.tokens_saved > 0 or report.messages_removed > 0

    def test_priority_ordering(self):
        from app.services.token_budget_manager import MessagePriority

        # Verify priority ordering
        assert MessagePriority.SNAPSHOT < MessagePriority.TOOL_RESULT
        assert MessagePriority.TOOL_RESULT < MessagePriority.HISTORY_OLD
        assert MessagePriority.HISTORY_OLD < MessagePriority.SYSTEM_PROMPT
        assert MessagePriority.SYSTEM_PROMPT < MessagePriority.CURRENT_QUERY

    def test_model_token_limits(self):
        from app.services.token_budget_manager import get_model_token_limit

        assert get_model_token_limit("gpt-4o") == 128_000
        assert get_model_token_limit("claude-3-5-sonnet-20241022") == 200_000
        assert get_model_token_limit("gemini-1.5-pro") == 2_000_000
        # Unknown model fallback
        assert get_model_token_limit("unknown-model") == 32_000

    def test_smart_truncate_text(self):
        from app.services.token_budget_manager import TokenBudgetManager

        manager = TokenBudgetManager(max_context_tokens=10_000)

        long_text = (
            "Pierwszy akapit. " * 50
            + "\n\n"
            + "Drugi akapit. " * 50
            + "\n\n"
            + "Trzeci akapit. " * 50
            + "\n\n"
            + "Czwarty akapit. " * 50
        )
        truncated = manager._smart_truncate_text(long_text, 20)
        # Should contain truncation marker and be shorter than original
        assert "skrócona" in truncated or "pominięto" in truncated
        assert len(truncated) < len(long_text)


# ---------------------------------------------------------------------------
# Model Router Tests
# ---------------------------------------------------------------------------


class TestModelRouter:
    """Tests for enhanced ModelRouter."""

    def _make_router(self):
        from app.services.model_router import ModelRouter

        return ModelRouter()

    def test_classify_easy(self):
        router = self._make_router()
        assert router.classify_difficulty("cześć") == "easy"
        assert router.classify_difficulty("ok") == "easy"

    def test_classify_medium(self):
        router = self._make_router()
        assert router.classify_difficulty("jak działa Python?") == "medium"
        assert router.classify_difficulty("napisz mi funkcję") == "medium"

    def test_classify_hard(self):
        from app.services.model_router import DifficultyLevel

        router = self._make_router()
        result = router.classify_difficulty(
            "Przeanalizuj szczegółowo architekturę tego systemu, porównaj z alternatywami, "
            "zaprojektuj nowe rozwiązanie i zoptymalizuj algorytm sortowania pod kątem złożoności. "
            "Wyjaśnij szczegółowo krok po kroku jak zaimplementować strategię refaktoryzacji."
        )
        assert result == DifficultyLevel.HARD

    def test_analyze_query_intents(self):
        from app.services.model_router import QueryIntent

        router = self._make_router()

        # Code intent
        analysis = router.analyze_query("napisz funkcję w Python")
        assert QueryIntent.CODE in analysis.intents

        # Math intent
        analysis = router.analyze_query("oblicz 15% z 200")
        assert QueryIntent.MATH in analysis.intents

        # Search intent
        analysis = router.analyze_query("znajdź najnowsze informacje o AI")
        assert QueryIntent.SEARCH in analysis.intents

        # Document intent
        analysis = router.analyze_query("pokaż treść mojego dokumentu PDF")
        assert QueryIntent.DOCUMENT in analysis.intents

        # Memory intent
        analysis = router.analyze_query("zapamiętaj moje imię: Jan")
        assert QueryIntent.MEMORY in analysis.intents

        # Temporal intent
        analysis = router.analyze_query("jaka jest dzisiaj data?")
        assert QueryIntent.TEMPORAL in analysis.intents

        # Conversational intent
        analysis = router.analyze_query("cześć")
        assert QueryIntent.CONVERSATIONAL in analysis.intents

    def test_tool_recommendations(self):
        router = self._make_router()

        # Search query should recommend web_search
        recs = router.get_recommended_tools("znajdź aktualną cenę Bitcoin")
        tool_names = [r.tool_name for r in recs]
        assert "web_search" in tool_names

        # Math query should recommend calculate
        recs = router.get_recommended_tools("oblicz 2^10 + sqrt(144)")
        tool_names = [r.tool_name for r in recs]
        assert "calculate" in tool_names

        # Document query should recommend rag_search
        recs = router.get_recommended_tools("pokaż treść mojego dokumentu PDF")
        tool_names = [r.tool_name for r in recs]
        assert "rag_search" in tool_names

        # Time query should recommend get_datetime
        recs = router.get_recommended_tools("jaka jest teraz godzina?")
        tool_names = [r.tool_name for r in recs]
        assert "get_datetime" in tool_names

    def test_tool_recommendations_sorted_by_relevance(self):
        router = self._make_router()
        recs = router.get_recommended_tools("znajdź aktualną cenę Bitcoin")
        if len(recs) > 1:
            for i in range(len(recs) - 1):
                assert recs[i].relevance_score >= recs[i + 1].relevance_score

    def test_select_profile_eco(self):
        from app.services.model_router import DifficultyLevel, Profile

        router = self._make_router()
        assert router.select_profile(DifficultyLevel.EASY) == Profile.ECO

    def test_select_profile_smart(self):
        from app.services.model_router import DifficultyLevel, Profile

        router = self._make_router()
        assert router.select_profile(DifficultyLevel.MEDIUM) == Profile.SMART

    def test_select_profile_deep(self):
        from app.services.model_router import DifficultyLevel, Profile

        router = self._make_router()
        assert router.select_profile(DifficultyLevel.HARD, user_role="FULL_ACCESS") == Profile.DEEP

    def test_select_profile_demo_capped(self):
        from app.services.model_router import DifficultyLevel, Profile

        router = self._make_router()
        # DEMO users can't get DEEP
        assert router.select_profile(DifficultyLevel.HARD, user_role="DEMO") == Profile.SMART

    def test_select_profile_override(self):
        from app.services.model_router import DifficultyLevel, Profile

        router = self._make_router()
        assert router.select_profile(DifficultyLevel.EASY, user_mode="smart") == Profile.SMART

    def test_estimate_cost(self):
        from app.services.model_router import Profile

        router = self._make_router()
        estimate = router.estimate_cost(Profile.SMART, "openai", 1000, 500)
        assert estimate.estimated_cost_usd > 0
        assert estimate.provider == "openai"

    def test_calculate_smart_credits(self):
        router = self._make_router()
        assert router.calculate_smart_credits(100) == 1
        assert router.calculate_smart_credits(500) == 1
        assert router.calculate_smart_credits(1000) == 2
        assert router.calculate_smart_credits(5000) == 4

    def test_needs_confirmation(self):
        from app.services.model_router import Profile

        router = self._make_router()
        assert router.needs_confirmation(Profile.DEEP, "FULL_ACCESS") is True
        assert router.needs_confirmation(Profile.DEEP, "ADMIN") is False
        assert router.needs_confirmation(Profile.SMART, "FULL_ACCESS") is False

    def test_query_analysis_confidence(self):
        router = self._make_router()
        # Clear intent should have higher confidence
        analysis = router.analyze_query("przeanalizuj szczegółowo architekturę systemu")
        assert analysis.confidence > 0.5

    def test_query_analysis_signals(self):
        router = self._make_router()
        analysis = router.analyze_query("jak działa Python?")
        assert "word_count" in analysis.signals
        assert "structural_complexity" in analysis.signals
        assert "difficulty_score" in analysis.signals


# ---------------------------------------------------------------------------
# ToolResult Tests
# ---------------------------------------------------------------------------


class TestToolResult:
    """Tests for ToolResult message formatting."""

    def test_success_string(self):
        from app.tools.tool_registry import ToolResult

        result = ToolResult(success=True, data="Hello world")
        assert result.to_message_content() == "Hello world"

    def test_success_list(self):
        from app.tools.tool_registry import ToolResult

        result = ToolResult(
            success=True,
            data=[
                {"title": "Result 1", "snippet": "First result"},
                {"title": "Result 2", "snippet": "Second result"},
            ],
        )
        content = result.to_message_content()
        assert "Result 1" in content
        assert "Result 2" in content

    def test_success_dict(self):
        from app.tools.tool_registry import ToolResult

        result = ToolResult(success=True, data={"key": "value", "count": 42})
        content = result.to_message_content()
        assert "key: value" in content
        assert "count: 42" in content

    def test_success_none(self):
        from app.tools.tool_registry import ToolResult

        result = ToolResult(success=True, data=None)
        assert "pomyślnie" in result.to_message_content()

    def test_error(self):
        from app.tools.tool_registry import ToolResult

        result = ToolResult(success=False, error="Something went wrong", tool_name="test")
        content = result.to_message_content()
        assert "BŁĄD" in content
        assert "Something went wrong" in content

    def test_empty_list(self):
        from app.tools.tool_registry import ToolResult

        result = ToolResult(success=True, data=[])
        assert "Brak wyników" in result.to_message_content()


# ---------------------------------------------------------------------------
# Default Tools Factory Tests
# ---------------------------------------------------------------------------


class TestDefaultToolsFactory:
    """Tests for create_default_tools factory."""

    def test_creates_all_tools(self):
        from app.tools.tool_registry import create_default_tools

        registry = create_default_tools()
        tool_names = registry.list_tool_names(enabled_only=False)

        expected_tools = [
            "web_search",
            "vertex_search",
            "rag_search",
            "memory_read",
            "memory_write",
            "calculate",
            "get_datetime",
        ]
        for name in expected_tools:
            assert name in tool_names, f"Missing tool: {name}"

    def test_tool_categories(self):
        from app.tools.tool_registry import create_default_tools

        registry = create_default_tools()

        search_tools = registry.list_tools(category="search")
        assert len(search_tools) >= 2  # web_search, vertex_search, rag_search

        utility_tools = registry.list_tools(category="utility")
        assert len(utility_tools) >= 1  # calculate, get_datetime

        memory_tools = registry.list_tools(category="memory")
        assert len(memory_tools) >= 2  # memory_read, memory_write

    @pytest.mark.asyncio
    async def test_calculate_tool(self):
        from app.tools.tool_registry import create_default_tools

        registry = create_default_tools()
        result = await registry.execute("calculate", {"expression": "2**10"})
        assert result.success is True
        assert "1024" in result.data

    @pytest.mark.asyncio
    async def test_calculate_tool_error(self):
        from app.tools.tool_registry import create_default_tools

        registry = create_default_tools()
        result = await registry.execute("calculate", {"expression": "invalid_expr()"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_get_datetime_tool(self):
        from app.tools.tool_registry import create_default_tools

        registry = create_default_tools()
        result = await registry.execute("get_datetime", {})
        assert result.success is True
        assert "UTC" in result.data


# ---------------------------------------------------------------------------
# Orchestrator Data Model Tests
# ---------------------------------------------------------------------------


class TestOrchestratorModels:
    """Tests for Orchestrator data models."""

    def test_orchestrator_request(self):
        from app.services.orchestrator import OrchestratorRequest

        user = MagicMock()
        user.telegram_id = 12345
        user.role = "DEMO"
        user.default_mode = "eco"

        req = OrchestratorRequest(user=user, query="Hello")
        assert req.query == "Hello"
        assert req.session_id is None
        assert req.deep_confirmed is False

    def test_orchestrator_response(self):
        from app.services.orchestrator import OrchestratorResponse

        resp = OrchestratorResponse(
            content="Test response",
            provider="openai",
            model="gpt-4o",
            profile="smart",
            difficulty="medium",
            cost_usd=0.001,
            latency_ms=500,
            input_tokens=100,
            output_tokens=50,
            fallback_used=False,
        )
        assert resp.content == "Test response"
        assert resp.react_iterations == 0
        assert resp.tools_used == []

    def test_thought_step(self):
        from app.services.orchestrator import AgentAction, ThoughtStep

        step = ThoughtStep(
            iteration=1,
            action=AgentAction.USE_TOOL,
            thought="Need to search for information",
            tool_name="web_search",
            tool_args={"query": "test"},
        )
        assert step.action == AgentAction.USE_TOOL
        assert step.tool_name == "web_search"


# ---------------------------------------------------------------------------
# Integration-style Tests (without DB)
# ---------------------------------------------------------------------------


class TestToolRegistrySchemaConsistency:
    """Test that all provider schemas are consistent."""

    def test_all_providers_generate_valid_schemas(self):
        from app.tools.tool_registry import create_default_tools

        registry = create_default_tools()

        providers = ["openai", "claude", "gemini", "deepseek", "groq", "grok", "openrouter"]

        for provider in providers:
            schemas = registry.get_tools_for_provider(provider)
            assert len(schemas) > 0, f"No schemas for provider: {provider}"

            for schema in schemas:
                # All schemas must have a name
                if provider == "gemini":
                    assert "name" in schema
                    assert "description" in schema
                    assert "parameters" in schema
                elif provider == "claude":
                    assert "name" in schema
                    assert "description" in schema
                    assert "input_schema" in schema
                else:  # OpenAI-style
                    assert "type" in schema
                    assert schema["type"] == "function"
                    assert "function" in schema
                    assert "name" in schema["function"]

    def test_schema_parameter_completeness(self):
        from app.tools.tool_registry import create_default_tools

        registry = create_default_tools()

        for tool in registry.list_tools():
            # OpenAI schema
            openai_schema = tool.to_openai_schema()
            props = openai_schema["function"]["parameters"]["properties"]
            required = openai_schema["function"]["parameters"]["required"]

            for param in tool.parameters:
                assert (
                    param.name in props
                ), f"Missing param {param.name} in OpenAI schema for {tool.name}"
                if param.required:
                    assert (
                        param.name in required
                    ), f"Required param {param.name} not in required list for {tool.name}"
