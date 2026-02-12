"""
Web search tool using Brave Search API.
"""

from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import ToolError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class WebSearchTool:
    """Tool for web search using Brave Search API."""

    BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize web search tool.

        Args:
            api_key: Brave Search API key
        """
        self.api_key = api_key or settings.brave_search_api_key

    def is_available(self) -> bool:
        """
        Check if web search is available.

        Returns:
            True if API key configured
        """
        return self.api_key is not None and len(self.api_key) > 0

    async def search(
        self, query: str, max_results: int = 5
    ) -> list[dict[str, Any]]:
        """
        Perform web search.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search result dicts with title, snippet, url

        Raises:
            ToolError: If search fails
        """
        if not self.is_available():
            logger.warning("Web search not available (missing API key)")
            return []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.BRAVE_SEARCH_URL,
                    params={"q": query, "count": max_results},
                    headers={"X-Subscription-Token": self.api_key},
                    timeout=10.0,
                )

                response.raise_for_status()
                data = response.json()

            # Parse results
            results = []
            for item in data.get("web", {}).get("results", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("description", ""),
                        "url": item.get("url", ""),
                        "source": "web",
                    }
                )

            logger.info(f"Web search returned {len(results)} results for: {query[:50]}")

            return results

        except httpx.HTTPStatusError as e:
            logger.error(f"Web search HTTP error: {e.response.status_code}")
            raise ToolError(
                f"Web search failed: HTTP {e.response.status_code}",
                {"query": query, "tool": "web_search"},
            )
        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            raise ToolError(
                f"Web search failed: {str(e)}",
                {"query": query, "tool": "web_search"},
            )
