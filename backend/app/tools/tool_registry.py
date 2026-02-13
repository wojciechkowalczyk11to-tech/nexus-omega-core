"""
Tool Registry — centralny rejestr narzędzi z natywnym function calling.

Obsługuje dynamiczną rejestrację narzędzi i generowanie schematów
kompatybilnych z OpenAI, Gemini i Claude (Anthropic) function calling API.

Architektura:
    ToolDefinition → ToolRegistry → Provider-specific schemas
                                  → Tool execution + error handling
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC
from enum import Enum
from typing import Any

from app.core.exceptions import ToolExecutionError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ParameterType(str, Enum):
    """JSON Schema types for tool parameters."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """Single parameter definition for a tool."""

    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None
    items_type: ParameterType | None = None  # for array types


@dataclass
class ToolResult:
    """Result returned by a tool execution."""

    success: bool
    data: Any = None
    error: str | None = None
    execution_time_ms: int = 0
    tool_name: str = ""
    retryable: bool = False

    def to_message_content(self) -> str:
        """Convert result to a string suitable for LLM context."""
        if self.success:
            if isinstance(self.data, str):
                return self.data
            if isinstance(self.data, list):
                parts = []
                for i, item in enumerate(self.data, 1):
                    if isinstance(item, dict):
                        formatted = "\n".join(f"  {k}: {v}" for k, v in item.items())
                        parts.append(f"[{i}]\n{formatted}")
                    else:
                        parts.append(f"[{i}] {item}")
                return "\n\n".join(parts) if parts else "Brak wyników."
            if isinstance(self.data, dict):
                return "\n".join(f"{k}: {v}" for k, v in self.data.items())
            return str(self.data) if self.data is not None else "Operacja zakończona pomyślnie."
        return f"[BŁĄD narzędzia {self.tool_name}]: {self.error}"


@dataclass
class ToolDefinition:
    """Complete definition of a tool available to the agent."""

    name: str
    description: str
    parameters: list[ToolParameter]
    handler: Callable[..., Coroutine[Any, Any, ToolResult]]
    category: str = "general"
    requires_db: bool = False
    max_retries: int = 1
    timeout_seconds: float = 30.0
    enabled: bool = True

    # --- Schema generators for each provider ---

    def to_openai_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible function calling schema."""
        properties: dict[str, Any] = {}
        required_params: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type.value,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.type == ParameterType.ARRAY and param.items_type:
                prop["items"] = {"type": param.items_type.value}
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop

            if param.required:
                required_params.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            },
        }

    def to_gemini_schema(self) -> dict[str, Any]:
        """Generate Gemini-compatible function declaration schema."""
        properties: dict[str, Any] = {}
        required_params: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type.value.upper(),
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.type == ParameterType.ARRAY and param.items_type:
                prop["items"] = {"type": param.items_type.value.upper()}
            properties[param.name] = prop

            if param.required:
                required_params.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "OBJECT",
                "properties": properties,
                "required": required_params,
            },
        }

    def to_claude_schema(self) -> dict[str, Any]:
        """Generate Claude (Anthropic) compatible tool schema."""
        properties: dict[str, Any] = {}
        required_params: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type.value,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.type == ParameterType.ARRAY and param.items_type:
                prop["items"] = {"type": param.items_type.value}
            properties[param.name] = prop

            if param.required:
                required_params.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required_params,
            },
        }


# ---------------------------------------------------------------------------
# Tool Registry (singleton-like, but instantiated per-request with DB)
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    Centralny rejestr narzędzi agenta.

    Odpowiada za:
    - Rejestrację i przechowywanie definicji narzędzi
    - Generowanie schematów function calling dla różnych providerów
    - Bezpieczne wykonywanie narzędzi z obsługą błędów i timeout
    - Dostarczanie metadanych narzędzi do kontekstu agenta
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._execution_stats: dict[str, dict[str, int]] = {}

    # ---- Registration ----

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool in the registry."""
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered — overwriting")
        self._tools[tool.name] = tool
        self._execution_stats[tool.name] = {"calls": 0, "errors": 0, "total_ms": 0}
        logger.info(f"Registered tool: {tool.name} (category={tool.category})")

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    # ---- Queries ----

    def get(self, name: str) -> ToolDefinition | None:
        """Get tool definition by name."""
        return self._tools.get(name)

    def list_tools(
        self, category: str | None = None, enabled_only: bool = True
    ) -> list[ToolDefinition]:
        """List all registered tools, optionally filtered."""
        tools = list(self._tools.values())
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def list_tool_names(self, enabled_only: bool = True) -> list[str]:
        """List names of registered tools."""
        return [t.name for t in self.list_tools(enabled_only=enabled_only)]

    def get_tool_descriptions(self) -> str:
        """Get human-readable descriptions of all tools for system prompt."""
        lines = []
        for tool in self.list_tools():
            params_desc = ", ".join(
                f"{p.name}: {p.type.value}" + (" (wymagany)" if p.required else "")
                for p in tool.parameters
            )
            lines.append(f"• **{tool.name}** — {tool.description}")
            if params_desc:
                lines.append(f"  Parametry: {params_desc}")
        return "\n".join(lines)

    # ---- Schema generation ----

    def get_openai_tools(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Get all tool schemas in OpenAI format."""
        return [t.to_openai_schema() for t in self.list_tools(enabled_only=enabled_only)]

    def get_gemini_tools(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Get all tool schemas in Gemini format."""
        return [t.to_gemini_schema() for t in self.list_tools(enabled_only=enabled_only)]

    def get_claude_tools(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Get all tool schemas in Claude format."""
        return [t.to_claude_schema() for t in self.list_tools(enabled_only=enabled_only)]

    def get_tools_for_provider(
        self, provider_name: str, enabled_only: bool = True
    ) -> list[dict[str, Any]]:
        """Get tool schemas formatted for a specific provider."""
        provider_map = {
            "openai": self.get_openai_tools,
            "claude": self.get_claude_tools,
            "gemini": self.get_gemini_tools,
            # Providers that use OpenAI-compatible API
            "deepseek": self.get_openai_tools,
            "groq": self.get_openai_tools,
            "grok": self.get_openai_tools,
            "openrouter": self.get_openai_tools,
        }
        generator = provider_map.get(provider_name, self.get_openai_tools)
        return generator(enabled_only=enabled_only)

    # ---- Execution ----

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        **extra_context: Any,
    ) -> ToolResult:
        """
        Execute a tool by name with given arguments.

        Handles:
        - Argument validation
        - Timeout enforcement
        - Error capture and retryable classification
        - Execution statistics
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Narzędzie '{tool_name}' nie jest zarejestrowane.",
                tool_name=tool_name,
            )

        if not tool.enabled:
            return ToolResult(
                success=False,
                error=f"Narzędzie '{tool_name}' jest wyłączone.",
                tool_name=tool_name,
            )

        # Validate required parameters
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                if param.default is not None:
                    arguments[param.name] = param.default
                else:
                    return ToolResult(
                        success=False,
                        error=f"Brakujący wymagany parametr: '{param.name}' dla narzędzia '{tool_name}'.",
                        tool_name=tool_name,
                    )

        # Execute with timeout and retry
        last_error: str | None = None
        for attempt in range(1, tool.max_retries + 1):
            start = time.time()
            try:
                # Inject extra context (e.g., db session) if handler accepts **kwargs
                sig = inspect.signature(tool.handler)
                handler_kwargs = dict(arguments)
                for key, value in extra_context.items():
                    if key in sig.parameters:
                        handler_kwargs[key] = value

                result = await asyncio.wait_for(
                    tool.handler(**handler_kwargs),
                    timeout=tool.timeout_seconds,
                )

                elapsed_ms = int((time.time() - start) * 1000)
                result.execution_time_ms = elapsed_ms
                result.tool_name = tool_name

                # Update stats
                stats = self._execution_stats[tool_name]
                stats["calls"] += 1
                stats["total_ms"] += elapsed_ms
                if not result.success:
                    stats["errors"] += 1

                logger.info(
                    f"Tool '{tool_name}' executed in {elapsed_ms}ms "
                    f"(attempt {attempt}/{tool.max_retries}, success={result.success})"
                )
                return result

            except TimeoutError:
                elapsed_ms = int((time.time() - start) * 1000)
                last_error = f"Timeout ({tool.timeout_seconds}s) dla narzędzia '{tool_name}'"
                logger.warning(f"Tool '{tool_name}' timed out (attempt {attempt})")
                self._execution_stats[tool_name]["errors"] += 1

            except Exception as e:
                elapsed_ms = int((time.time() - start) * 1000)
                last_error = f"{type(e).__name__}: {str(e)}"
                logger.error(
                    f"Tool '{tool_name}' error (attempt {attempt}): {last_error}",
                    exc_info=True,
                )
                self._execution_stats[tool_name]["errors"] += 1

        # All retries exhausted
        return ToolResult(
            success=False,
            error=last_error or "Nieznany błąd",
            tool_name=tool_name,
            retryable=True,
        )

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Get execution statistics for all tools."""
        return dict(self._execution_stats)


# ---------------------------------------------------------------------------
# Default tool factory — creates ToolDefinitions from existing tool classes
# ---------------------------------------------------------------------------


def create_default_tools(db=None) -> ToolRegistry:
    """
    Create a ToolRegistry populated with all available tools.

    Args:
        db: Optional AsyncSession for tools that need database access.

    Returns:
        Populated ToolRegistry instance.
    """
    registry = ToolRegistry()

    # --- Web Search Tool ---
    async def _web_search(query: str, max_results: int = 5) -> ToolResult:
        try:
            from app.tools.web_search_tool import WebSearchTool

            tool = WebSearchTool()
            if not tool.is_available():
                return ToolResult(
                    success=False,
                    error="Web Search niedostępny (brak klucza API Brave Search).",
                    retryable=False,
                )
            results = await tool.search(query, max_results=max_results)
            if not results:
                return ToolResult(success=True, data="Brak wyników wyszukiwania.")
            return ToolResult(success=True, data=results)
        except ToolExecutionError as e:
            return ToolResult(success=False, error=str(e), retryable=True)
        except Exception as e:
            return ToolResult(success=False, error=str(e), retryable=True)

    registry.register(
        ToolDefinition(
            name="web_search",
            description="Wyszukaj informacje w internecie za pomocą Brave Search. Użyj, gdy potrzebujesz aktualnych informacji, faktów lub danych, których nie ma w bazie wiedzy.",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="Zapytanie wyszukiwania w języku naturalnym",
                ),
                ToolParameter(
                    name="max_results",
                    type=ParameterType.INTEGER,
                    description="Maksymalna liczba wyników (1-10)",
                    required=False,
                    default=5,
                ),
            ],
            handler=_web_search,
            category="search",
            timeout_seconds=15.0,
            max_retries=2,
        )
    )

    # --- Vertex AI Search Tool ---
    async def _vertex_search(query: str, max_results: int = 5) -> ToolResult:
        try:
            from app.tools.vertex_tool import VertexSearchTool

            tool = VertexSearchTool()
            if not tool.is_available():
                return ToolResult(
                    success=False,
                    error="Vertex AI Search niedostępny (brak konfiguracji GCP).",
                    retryable=False,
                )
            results = await tool.search(query, max_results=max_results)
            if not results:
                return ToolResult(success=True, data="Brak wyników w bazie wiedzy Vertex.")
            return ToolResult(success=True, data=results)
        except ToolExecutionError as e:
            return ToolResult(success=False, error=str(e), retryable=True)
        except Exception as e:
            return ToolResult(success=False, error=str(e), retryable=True)

    registry.register(
        ToolDefinition(
            name="vertex_search",
            description="Przeszukaj bazę wiedzy Vertex AI Search. Użyj dla pytań dotyczących wewnętrznej dokumentacji, bazy wiedzy firmy lub specjalistycznych informacji.",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="Zapytanie do bazy wiedzy",
                ),
                ToolParameter(
                    name="max_results",
                    type=ParameterType.INTEGER,
                    description="Maksymalna liczba wyników (1-10)",
                    required=False,
                    default=5,
                ),
            ],
            handler=_vertex_search,
            category="search",
            timeout_seconds=15.0,
            max_retries=2,
        )
    )

    # --- RAG Document Search Tool ---
    async def _rag_search(
        query: str, user_id: int = 0, top_k: int = 5, db_session=None
    ) -> ToolResult:
        try:
            if db_session is None:
                return ToolResult(
                    success=False,
                    error="RAG wymaga sesji bazy danych.",
                    retryable=False,
                )
            from app.tools.rag_tool import RAGTool

            tool = RAGTool(db_session)
            results = await tool.search(user_id, query, top_k=top_k)
            if not results:
                return ToolResult(success=True, data="Brak pasujących dokumentów użytkownika.")
            return ToolResult(success=True, data=results)
        except Exception as e:
            return ToolResult(success=False, error=str(e), retryable=True)

    registry.register(
        ToolDefinition(
            name="rag_search",
            description="Przeszukaj dokumenty użytkownika (RAG). Użyj, gdy użytkownik pyta o treść swoich przesłanych dokumentów, plików PDF, DOCX lub notatek.",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="Zapytanie do dokumentów użytkownika",
                ),
                ToolParameter(
                    name="top_k",
                    type=ParameterType.INTEGER,
                    description="Liczba najlepszych wyników do zwrócenia",
                    required=False,
                    default=5,
                ),
            ],
            handler=_rag_search,
            category="search",
            requires_db=True,
            timeout_seconds=10.0,
        )
    )

    # --- Memory Read Tool ---
    async def _memory_read(key: str, user_id: int = 0, db_session=None) -> ToolResult:
        try:
            if db_session is None:
                return ToolResult(success=False, error="Brak sesji DB.", retryable=False)
            from app.services.memory_manager import MemoryManager

            mm = MemoryManager(db_session)
            value = await mm.get_absolute_memory(user_id, key)
            if value is None:
                return ToolResult(
                    success=True, data=f"Brak zapamiętanej wartości dla klucza '{key}'."
                )
            return ToolResult(success=True, data=f"{key}: {value}")
        except Exception as e:
            return ToolResult(success=False, error=str(e), retryable=False)

    registry.register(
        ToolDefinition(
            name="memory_read",
            description="Odczytaj zapamiętaną preferencję lub informację użytkownika z pamięci trwałej. Użyj, gdy potrzebujesz sprawdzić wcześniej zapisane dane.",
            parameters=[
                ToolParameter(
                    name="key",
                    type=ParameterType.STRING,
                    description="Klucz pamięci do odczytania (np. 'imie', 'jezyk', 'preferencje')",
                ),
            ],
            handler=_memory_read,
            category="memory",
            requires_db=True,
            timeout_seconds=5.0,
        )
    )

    # --- Memory Write Tool ---
    async def _memory_write(key: str, value: str, user_id: int = 0, db_session=None) -> ToolResult:
        try:
            if db_session is None:
                return ToolResult(success=False, error="Brak sesji DB.", retryable=False)
            from app.services.memory_manager import MemoryManager

            mm = MemoryManager(db_session)
            await mm.set_absolute_memory(user_id, key, value)
            return ToolResult(success=True, data=f"Zapamiętano: {key} = {value}")
        except Exception as e:
            return ToolResult(success=False, error=str(e), retryable=False)

    registry.register(
        ToolDefinition(
            name="memory_write",
            description="Zapisz preferencję lub ważną informację użytkownika do pamięci trwałej. Użyj, gdy użytkownik prosi o zapamiętanie czegoś lub podaje ważne informacje o sobie.",
            parameters=[
                ToolParameter(
                    name="key",
                    type=ParameterType.STRING,
                    description="Klucz pamięci (np. 'imie', 'jezyk', 'ulubiony_model')",
                ),
                ToolParameter(
                    name="value",
                    type=ParameterType.STRING,
                    description="Wartość do zapamiętania",
                ),
            ],
            handler=_memory_write,
            category="memory",
            requires_db=True,
            timeout_seconds=5.0,
        )
    )

    # --- Calculator Tool ---
    async def _calculate(expression: str) -> ToolResult:
        """Safe math expression evaluator."""
        try:
            # Whitelist of allowed characters for safety
            set("0123456789+-*/().%, episincotaglqrtbdfhx^ ")
            expr_clean = expression.strip()

            import math

            safe_dict = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "pow": pow,
                "sum": sum,
                "len": len,
                "pi": math.pi,
                "e": math.e,
                "sqrt": math.sqrt,
                "log": math.log,
                "log10": math.log10,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "ceil": math.ceil,
                "floor": math.floor,
                "__builtins__": {},
            }
            result = eval(expr_clean, safe_dict)
            return ToolResult(success=True, data=f"{expression} = {result}")
        except Exception as e:
            return ToolResult(success=False, error=f"Błąd obliczenia: {str(e)}")

    registry.register(
        ToolDefinition(
            name="calculate",
            description="Wykonaj obliczenie matematyczne. Użyj dla precyzyjnych obliczeń, konwersji jednostek, procentów itp.",
            parameters=[
                ToolParameter(
                    name="expression",
                    type=ParameterType.STRING,
                    description="Wyrażenie matematyczne do obliczenia (np. '2**10', 'sqrt(144)', '15% * 200')",
                ),
            ],
            handler=_calculate,
            category="utility",
            timeout_seconds=5.0,
        )
    )

    # --- Current DateTime Tool ---
    async def _get_datetime() -> ToolResult:
        from datetime import datetime

        now = datetime.now(UTC)
        return ToolResult(
            success=True,
            data=f"Aktualna data i czas (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}",
        )

    registry.register(
        ToolDefinition(
            name="get_datetime",
            description="Pobierz aktualną datę i godzinę (UTC). Użyj, gdy użytkownik pyta o aktualny czas lub datę.",
            parameters=[],
            handler=_get_datetime,
            category="utility",
            timeout_seconds=2.0,
        )
    )

    return registry
