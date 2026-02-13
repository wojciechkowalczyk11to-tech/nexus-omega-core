"""
Vertex AI Search tool for knowledge base queries with citations.
"""

from typing import Any

from google.cloud import discoveryengine_v1 as discoveryengine

from app.core.config import settings
from app.core.exceptions import ToolExecutionError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class VertexSearchTool:
    """Tool for Vertex AI Search integration."""

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "global",
        data_store_id: str | None = None,
    ) -> None:
        """
        Initialize Vertex AI Search tool.

        Args:
            project_id: GCP project ID
            location: GCP location
            data_store_id: Vertex AI Search data store ID
        """
        self.project_id = project_id or settings.vertex_project_id
        self.location = location
        self.data_store_id = data_store_id or settings.vertex_search_datastore_id

        if not self.project_id or not self.data_store_id:
            logger.warning("Vertex AI Search not configured (missing project_id or data_store_id)")
            self.client = None
        else:
            try:
                self.client = discoveryengine.SearchServiceClient()
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI Search client: {e}")
                self.client = None

    def is_available(self) -> bool:
        """
        Check if Vertex AI Search is available.

        Returns:
            True if configured and client initialized
        """
        return self.client is not None

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """
        Search Vertex AI knowledge base.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search result dicts with title, snippet, link, score

        Raises:
            ToolError: If search fails
        """
        if not self.is_available():
            logger.warning("Vertex AI Search not available, skipping")
            return []

        try:
            # Build search request
            serving_config = (
                f"projects/{self.project_id}/locations/{self.location}/"
                f"collections/default_collection/dataStores/{self.data_store_id}/"
                f"servingConfigs/default_config"
            )

            request = discoveryengine.SearchRequest(
                serving_config=serving_config,
                query=query,
                page_size=max_results,
            )

            # Execute search
            response = self.client.search(request)

            # Parse results
            results = []
            for result in response.results:
                doc = result.document

                # Extract metadata
                title = doc.derived_struct_data.get("title", "Untitled")
                snippet = doc.derived_struct_data.get("snippets", [{}])[0].get("snippet", "")
                link = doc.derived_struct_data.get("link", "")

                results.append(
                    {
                        "title": title,
                        "snippet": snippet,
                        "link": link,
                        "score": 0.9,  # Placeholder - Vertex doesn't expose scores directly
                        "source": "vertex",
                    }
                )

            logger.info(f"Vertex AI Search returned {len(results)} results for: {query[:50]}")

            return results

        except Exception as e:
            logger.error(f"Vertex AI Search error: {e}", exc_info=True)
            raise ToolExecutionError(
                "vertex",
                f"Vertex AI Search failed: {str(e)}",
                {"query": query},
            ) from e

    def format_citations(self, results: list[dict[str, Any]]) -> str:
        """
        Format search results as citations.

        Args:
            results: List of search result dicts

        Returns:
            Formatted citation string
        """
        if not results:
            return ""

        citations = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            link = result.get("link", "")
            snippet = result.get("snippet", "")[:100]

            citation = f"{i}. **{title}**"
            if link:
                citation += f" - {link}"
            if snippet:
                citation += f"\n   _{snippet}_"

            citations.append(citation)

        return "\n\n".join(citations)
