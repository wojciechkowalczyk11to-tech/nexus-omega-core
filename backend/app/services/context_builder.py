"""
Context Builder — budowanie kontekstu z priorytetami tokenów.

Zintegrowany z TokenBudgetManager dla inteligentnego zarządzania kontekstem.
Obsługuje budowanie kontekstu z wielu źródeł z priorytetyzacją.

Źródła kontekstu (w kolejności priorytetów):
1. System prompt (najwyższy priorytet)
2. Bieżące zapytanie użytkownika
3. Pamięć użytkownika
4. Wyniki Vertex AI Search
5. Wyniki RAG
6. Wyniki Web Search
7. Historia sesji (snapshot + ostatnie wiadomości)
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.services.memory_manager import MemoryManager
from app.services.token_budget_manager import (
    MessagePriority,
    PrioritizedMessage,
    TokenBudgetManager,
    TokenCounter,
)
from app.tools.rag_tool import RAGTool
from app.tools.vertex_tool import VertexSearchTool
from app.tools.web_search_tool import WebSearchTool

logger = get_logger(__name__)


class ContextBuilder:
    """Builder for AI context from multiple sources with token budgeting."""

    # System prompt template
    SYSTEM_PROMPT = """Jesteś NexusOmegaCore - zaawansowanym asystentem AI z dostępem do wielu źródeł wiedzy.

Twoje możliwości:
- Odpowiadasz po polsku, chyba że użytkownik poprosi o inny język
- Masz dostęp do bazy wiedzy (Vertex AI Search)
- Możesz wyszukiwać w internecie (Brave Search)
- Masz dostęp do dokumentów użytkownika (RAG)
- Pamiętasz kontekst konwersacji i preferencje użytkownika

Zasady:
- Zawsze cytuj źródła, jeśli korzystasz z zewnętrznych informacji
- Bądź precyzyjny i konkretny
- Przyznaj się, jeśli czegoś nie wiesz
- Formatuj odpowiedzi czytelnie (Markdown)
"""

    def __init__(self, db: AsyncSession) -> None:
        """
        Initialize context builder.

        Args:
            db: Database session
        """
        self.db = db
        self.memory_manager = MemoryManager(db)
        self.rag_tool = RAGTool(db)
        self.vertex_tool = VertexSearchTool()
        self.web_tool = WebSearchTool()

    async def build_context(
        self,
        user_id: int,
        session_id: int,
        query: str,
        use_vertex: bool = True,
        use_rag: bool = True,
        use_web: bool = False,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        """
        Build context messages for AI request.

        This is the backward-compatible interface that returns flat messages.
        For prioritized context, use build_prioritized_context().

        Args:
            user_id: User's Telegram ID
            session_id: Session ID
            query: User query
            use_vertex: Whether to use Vertex AI Search
            use_rag: Whether to use RAG documents
            use_web: Whether to use web search

        Returns:
            Tuple of (messages, sources)
        """
        prioritized, sources = await self.build_prioritized_context(
            user_id=user_id,
            session_id=session_id,
            query=query,
            use_vertex=use_vertex,
            use_rag=use_rag,
            use_web=use_web,
        )

        # Flatten to simple message list
        messages = [pm.message for pm in prioritized]

        logger.info(
            f"Built context: {len(messages)} messages, {len(sources)} sources, "
            f"~{TokenCounter.count_messages(messages)} tokens"
        )

        return messages, sources

    async def build_prioritized_context(
        self,
        user_id: int,
        session_id: int,
        query: str,
        use_vertex: bool = True,
        use_rag: bool = True,
        use_web: bool = False,
    ) -> tuple[list[PrioritizedMessage], list[dict[str, Any]]]:
        """
        Build context with priority metadata for token budgeting.

        Args:
            user_id: User's Telegram ID
            session_id: Session ID
            query: User query
            use_vertex: Whether to use Vertex AI Search
            use_rag: Whether to use RAG documents
            use_web: Whether to use web search

        Returns:
            Tuple of (prioritized messages, sources)
        """
        prioritized: list[PrioritizedMessage] = []
        sources: list[dict[str, Any]] = []

        # 1. System prompt (highest priority, never truncated)
        system_prompt = self.SYSTEM_PROMPT

        # 2. Add absolute user memory
        memories = await self.memory_manager.list_absolute_memories(user_id)
        if memories:
            memory_text = "\n".join(
                [f"- {mem.key}: {mem.value}" for mem in memories[:5]]
            )
            system_prompt += f"\n\n**Preferencje użytkownika:**\n{memory_text}"

        prioritized.append(PrioritizedMessage(
            message={"role": "system", "content": system_prompt},
            priority=MessagePriority.SYSTEM_PROMPT,
            truncatable=False,
            source="system_prompt",
        ))

        # 3. Vertex AI Search
        if use_vertex and self.vertex_tool.is_available():
            try:
                vertex_results = await self.vertex_tool.search(query, max_results=3)
                if vertex_results:
                    sources.extend(vertex_results)

                    vertex_context = "**Wyniki z bazy wiedzy:**\n\n"
                    for i, result in enumerate(vertex_results, 1):
                        vertex_context += f"{i}. {result['title']}\n{result['snippet'][:200]}\n\n"

                    prioritized.append(PrioritizedMessage(
                        message={"role": "system", "content": vertex_context},
                        priority=MessagePriority.VERTEX_RESULT,
                        truncatable=True,
                        min_tokens=80,
                        source="vertex_search",
                    ))
                    logger.info(f"Added {len(vertex_results)} Vertex results to context")
            except Exception as e:
                logger.warning(f"Vertex search failed: {e}")

        # 4. RAG documents
        if use_rag:
            try:
                rag_results = await self.rag_tool.search(user_id, query, top_k=3)
                if rag_results:
                    sources.extend(rag_results)

                    rag_context = "**Wyniki z Twoich dokumentów:**\n\n"
                    for i, result in enumerate(rag_results, 1):
                        rag_context += f"{i}. {result['filename']}\n{result['content'][:200]}\n\n"

                    prioritized.append(PrioritizedMessage(
                        message={"role": "system", "content": rag_context},
                        priority=MessagePriority.RAG_RESULT,
                        truncatable=True,
                        min_tokens=80,
                        source="rag_search",
                    ))
                    logger.info(f"Added {len(rag_results)} RAG results to context")
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")

        # 5. Web search (if enabled)
        if use_web and self.web_tool.is_available():
            try:
                web_results = await self.web_tool.search(query, max_results=3)
                if web_results:
                    sources.extend(web_results)

                    web_context = "**Wyniki z internetu:**\n\n"
                    for i, result in enumerate(web_results, 1):
                        web_context += f"{i}. {result['title']}\n{result['snippet'][:200]}\n{result['url']}\n\n"

                    prioritized.append(PrioritizedMessage(
                        message={"role": "system", "content": web_context},
                        priority=MessagePriority.WEB_RESULT,
                        truncatable=True,
                        min_tokens=80,
                        source="web_search",
                    ))
                    logger.info(f"Added {len(web_results)} web results to context")
            except Exception as e:
                logger.warning(f"Web search failed: {e}")

        # 6. Session history (snapshot + recent messages)
        history_messages = await self.memory_manager.get_context_messages(session_id)
        for i, msg in enumerate(history_messages):
            content = msg.get("content", "")
            is_snapshot = "[Podsumowanie poprzedniej konwersacji]" in content

            if is_snapshot:
                prioritized.append(PrioritizedMessage(
                    message=msg,
                    priority=MessagePriority.SNAPSHOT,
                    truncatable=True,
                    min_tokens=100,
                    source="snapshot",
                ))
            else:
                recency_boost = min(i * 2, 10)
                prioritized.append(PrioritizedMessage(
                    message=msg,
                    priority=MessagePriority.HISTORY_OLD + recency_boost,
                    truncatable=True,
                    min_tokens=30,
                    source=f"history_{i}",
                ))

        # 7. Add current query (highest priority, never truncated)
        prioritized.append(PrioritizedMessage(
            message={"role": "user", "content": query},
            priority=MessagePriority.CURRENT_QUERY,
            truncatable=False,
            source="current_query",
        ))

        logger.info(
            f"Built prioritized context: {len(prioritized)} messages, "
            f"{len(sources)} sources"
        )

        return prioritized, sources

    async def build_context_with_budget(
        self,
        user_id: int,
        session_id: int,
        query: str,
        model: str = "",
        provider: str = "",
        use_vertex: bool = True,
        use_rag: bool = True,
        use_web: bool = False,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]], Any]:
        """
        Build context with automatic token budget management.

        Args:
            user_id: User's Telegram ID
            session_id: Session ID
            query: User query
            model: Model name for budget calculation
            provider: Provider name for budget calculation
            use_vertex: Whether to use Vertex AI Search
            use_rag: Whether to use RAG documents
            use_web: Whether to use web search

        Returns:
            Tuple of (messages, sources, budget_report)
        """
        # Build prioritized context
        prioritized, sources = await self.build_prioritized_context(
            user_id=user_id,
            session_id=session_id,
            query=query,
            use_vertex=use_vertex,
            use_rag=use_rag,
            use_web=use_web,
        )

        # Apply token budget
        budget_manager = TokenBudgetManager(model=model, provider=provider)
        messages, budget_report = budget_manager.apply_budget(prioritized)

        logger.info(
            f"Context with budget: {budget_report.original_token_count} → "
            f"{budget_report.final_token_count} tokens "
            f"({budget_report.budget_utilization:.1%} utilization)"
        )

        return messages, sources, budget_report
