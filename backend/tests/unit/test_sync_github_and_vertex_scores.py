"""
Focused tests for GitHub sync task and Vertex score ordering.
"""

import os
import sys
import types

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "x")
os.environ.setdefault("DEMO_UNLOCK_CODE", "x")
os.environ.setdefault("BOOTSTRAP_ADMIN_CODE", "x")
os.environ.setdefault("JWT_SECRET_KEY", "x")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///tmp/test.db")
os.environ.setdefault("POSTGRES_PASSWORD", "x")

from app.tools.vertex_tool import VertexSearchTool
from app.workers import tasks as worker_tasks


def test_sync_github_repo_returns_indexed_file_count_and_cleans_tempdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """Task should clone/index and cleanup temp directory in finally."""

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def commit(self):
            return None

    class FakeGitHubDevinTool:
        def __init__(self, user_id: int, db) -> None:
            self.user_id = user_id
            self.db = db
            self.sandbox = types.SimpleNamespace(repos_dir="")

        async def clone_repository(self, repo_url: str) -> dict[str, str]:
            return {"repo_name": "demo-repo"}

        async def index_repository(self, repo_name: str) -> dict[str, int]:
            return {"files_indexed": 3}

    temp_repo_dir = tmp_path / "sync-repo"
    temp_repo_dir.mkdir()

    monkeypatch.setitem(
        sys.modules, "app.db.session", types.SimpleNamespace(AsyncSessionLocal=FakeSession)
    )
    monkeypatch.setitem(
        sys.modules,
        "app.tools.github_devin_tool",
        types.SimpleNamespace(GitHubDevinTool=FakeGitHubDevinTool),
    )
    monkeypatch.setattr(worker_tasks.tempfile, "mkdtemp", lambda *args, **kwargs: str(temp_repo_dir))

    # Handle both direct function and Celery task wrapper
    func = worker_tasks.sync_github_repo
    if hasattr(func, "run"):
        result = func.run(123, "https://github.com/example/repo")
    else:
        result = func(123, "https://github.com/example/repo")

    assert result == {
        "status": "success",
        "repo_url": "https://github.com/example/repo",
        "files_indexed": 3,
    }
    assert not temp_repo_dir.exists()


@pytest.mark.asyncio
async def test_vertex_search_uses_rank_based_scores() -> None:
    """Scores should decrease with result position."""

    class FakeDocument:
        def __init__(self, title: str) -> None:
            self.derived_struct_data = {
                "title": title,
                "snippets": [{"snippet": f"{title} snippet"}],
                "link": f"https://example.com/{title}",
            }

    class FakeResult:
        def __init__(self, title: str) -> None:
            self.document = FakeDocument(title)

    class FakeSearchResponse:
        def __init__(self) -> None:
            self.results = [FakeResult("one"), FakeResult("two"), FakeResult("three")]

    class FakeSearchClient:
        def search(self, request):  # noqa: ARG002
            return FakeSearchResponse()

    tool = VertexSearchTool(project_id="project", data_store_id="store")
    tool.client = FakeSearchClient()

    results = await tool.search("query", max_results=3)

    assert [result["score"] for result in results] == [1.0, 0.5, 0.3333]
