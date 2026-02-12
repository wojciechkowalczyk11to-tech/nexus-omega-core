"""
Context builder for Step 3 of orchestrator flow.

Builds context from:
1. System prompt
2. Absolute user memory
3. Vertex AI Search results
4. RAG document chunks
5. Session history (snapshot + recent messages)
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging_config import get_logger
from app.services.memory_manager import MemoryManager
from app.tools.rag_tool import RAGTool
from app.tools.vertex_tool import VertexSearchTool
from app.tools.web_search_tool import WebSearchTool

logger = get_logger(__name__)


class ContextBuilder:
    """Builder for AI context from multiple sources."""

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
        messages = []
        sources = []

        # 1. System prompt
        system_prompt = self.SYSTEM_PROMPT

        # 2. Add absolute user memory
        memories = await self.memory_manager.list_absolute_memories(user_id)
        if memories:
            memory_text = "\n".join(
                [f"- {mem.key}: {mem.value}" for mem in memories[:5]]
            )
            system_prompt += f"\n\n**Preferencje użytkownika:**\n{memory_text}"

        messages.append({"role": "system", "content": system_prompt})

        # 3. Vertex AI Search
        if use_vertex and self.vertex_tool.is_available():
            try:
                vertex_results = await self.vertex_tool.search(query, max_results=3)
                if vertex_results:
                    sources.extend(vertex_results)

                    # Add to context
                    vertex_context = "**Wyniki z bazy wiedzy:**\n\n"
                    for i, result in enumerate(vertex_results, 1):
                        vertex_context += f"{i}. {result['title']}\n{result['snippet'][:200]}\n\n"

                    messages.append({"role": "system", "content": vertex_context})
                    logger.info(f"Added {len(vertex_results)} Vertex results to context")
            except Exception as e:
                logger.warning(f"Vertex search failed: {e}")

        # 4. RAG documents
        if use_rag:
            try:
                rag_results = await self.rag_tool.search(user_id, query, top_k=3)
                if rag_results:
                    sources.extend(rag_results)

                    # Add to context
                    rag_context = "**Wyniki z Twoich dokumentów:**\n\n"
                    for i, result in enumerate(rag_results, 1):
                        rag_context += f"{i}. {result['filename']}\n{result['content'][:200]}\n\n"

                    messages.append({"role": "system", "content": rag_context})
                    logger.info(f"Added {len(rag_results)} RAG results to context")
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")

        # 5. Web search (if enabled)
        if use_web and self.web_tool.is_available():
            try:
                web_results = await self.web_tool.search(query, max_results=3)
                if web_results:
                    sources.extend(web_results)

                    # Add to context
                    web_context = "**Wyniki z internetu:**\n\n"
                    for i, result in enumerate(web_results, 1):
                        web_context += f"{i}. {result['title']}\n{result['snippet'][:200]}\n{result['url']}\n\n"

                    messages.append({"role": "system", "content": web_context})
                    logger.info(f"Added {len(web_results)} web results to context")
            except Exception as e:
                logger.warning(f"Web search failed: {e}")

        # 6. Session history (snapshot + recent messages)
        history_messages = await self.memory_manager.get_context_messages(session_id)
        messages.extend(history_messages)

        # 7. Add current query
        messages.append({"role": "user", "content": query})

        logger.info(
            f"Built context: {len(messages)} messages, {len(sources)} sources"
        )

        return messages, sources
