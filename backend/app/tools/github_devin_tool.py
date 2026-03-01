"""
GitHub Devin-mode Tool — pełna integracja z GitHub w bezpiecznym sandboxie.

Funkcjonalności:
- Klonowanie repozytoriów do izolowanego sandbox
- Indeksowanie kodu z pgvector dla semantic search
- Operacje na plikach (read, write, edit)
- Tworzenie commitów i pull requestów
- Bezpieczne wykonywanie w sandboxie per-user

Architektura:
    User Request → Sandbox → Git Operations → GitHub API

Bezpieczeństwo:
- Wszystkie operacje w izolowanym sandboxie
- Walidacja ścieżek (path traversal protection)
- Limity rozmiaru plików i repozytoriów
- Timeout dla operacji git
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from git import Repo
from github import Github
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import GitHubError
from app.core.logging_config import get_logger
from app.db.models.rag_chunk import RagChunk
from app.db.models.rag_item import RagItem
from app.services.embedding_service import embed_texts
from app.services.sandbox import Sandbox

logger = get_logger(__name__)


class GitHubDevinTool:
    """
    GitHub Devin-mode tool with sandbox integration.

    Provides safe, isolated environment for GitHub operations.
    """

    # Supported code file extensions for indexing
    CODE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".sh",
        ".sql",
        ".html",
        ".css",
        ".scss",
        ".vue",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
    }

    # Repository size limits
    MAX_REPO_SIZE_MB = 500
    MAX_FILES_TO_INDEX = 1000

    def __init__(
        self,
        user_id: int,
        db: AsyncSession,
        github_token: str | None = None,
    ) -> None:
        """
        Initialize GitHub Devin tool.

        Args:
            user_id: User's Telegram ID
            db: Database session
            github_token: GitHub personal access token
        """
        self.user_id = user_id
        self.db = db
        self.sandbox = Sandbox(user_id)
        self.github_token = github_token or settings.github_token

        if self.github_token:
            self.github = Github(self.github_token)
        else:
            self.github = None
            logger.warning("GitHub token not provided, some features will be limited")

    async def clone_repository(
        self,
        repo_url: str,
        branch: str = "main",
    ) -> dict[str, Any]:
        """
        Clone GitHub repository to sandbox.

        Args:
            repo_url: Repository URL (e.g., https://github.com/user/repo)
            branch: Branch to clone

        Returns:
            Clone result with path and stats

        Raises:
            GitHubError: If clone fails
        """
        # Extract repo name from URL
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_path = self.sandbox.get_repos_path(repo_name)

        # Check if already cloned
        if os.path.exists(repo_path):
            logger.info("Repository %s already exists, pulling latest", repo_name)
            try:
                repo = Repo(repo_path)
                origin = repo.remotes.origin
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: origin.pull(),
                )
                return {
                    "repo_name": repo_name,
                    "repo_path": repo_path,
                    "status": "updated",
                    "branch": branch,
                }
            except Exception as e:
                logger.error("Failed to pull repository: %s", e)
                raise GitHubError(f"Failed to update repository: {str(e)}") from e

        try:
            # Clone repository
            logger.info("Cloning repository %s to %s", repo_url, repo_path)

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: Repo.clone_from(
                    repo_url,
                    repo_path,
                    branch=branch,
                    depth=1,  # Shallow clone for speed
                ),
            )

            # Check repo size
            repo_size_mb = self._get_directory_size(repo_path) / (1024 * 1024)
            if repo_size_mb > self.MAX_REPO_SIZE_MB:
                # Clean up
                import shutil

                shutil.rmtree(repo_path)
                raise GitHubError(
                    f"Repository too large: {repo_size_mb:.1f}MB (max {self.MAX_REPO_SIZE_MB}MB)"
                )

            logger.info("Successfully cloned %s (%sMB)", repo_name, repo_size_mb:.1f)

            return {
                "repo_name": repo_name,
                "repo_path": repo_path,
                "status": "cloned",
                "branch": branch,
                "size_mb": repo_size_mb,
            }

        except Exception as e:
            logger.error("Failed to clone repository: %s", e)
            raise GitHubError(f"Failed to clone repository: {str(e)}") from e

    async def index_repository(
        self,
        repo_name: str,
    ) -> dict[str, Any]:
        """
        Index repository code with pgvector embeddings.

        Args:
            repo_name: Repository name

        Returns:
            Indexing result with stats

        Raises:
            GitHubError: If indexing fails
        """
        repo_path = self.sandbox.get_repos_path(repo_name)

        if not os.path.exists(repo_path):
            raise GitHubError(f"Repository not found: {repo_name}")

        # Find all code files
        code_files = []
        for root, _dirs, files in os.walk(repo_path):
            # Skip .git directory
            if ".git" in root:
                continue

            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()

                if file_ext in self.CODE_EXTENSIONS:
                    code_files.append(file_path)

                    if len(code_files) >= self.MAX_FILES_TO_INDEX:
                        break

            if len(code_files) >= self.MAX_FILES_TO_INDEX:
                break

        if not code_files:
            return {
                "repo_name": repo_name,
                "files_indexed": 0,
                "chunks_created": 0,
            }

        logger.info("Indexing %s files from %s", len(code_files), repo_name)

        # Create RAG item for repository
        rag_item = RagItem(
            user_id=self.user_id,
            scope="user",
            source_type="github",
            source_url=f"file://{repo_path}",
            filename=f"{repo_name}_codebase",
            stored_path=repo_path,
            chunk_count=0,
            status="processing",
            item_metadata={
                "repo_name": repo_name,
                "files_count": len(code_files),
            },
        )

        self.db.add(rag_item)
        await self.db.flush()
        await self.db.refresh(rag_item)

        # Process files and create chunks
        all_chunks = []
        chunk_index = 0

        for file_path in code_files:
            try:
                # Read file content
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content.strip():
                    continue

                # Get relative path for metadata
                rel_path = os.path.relpath(file_path, repo_path)

                # Chunk file content (by function/class for code)
                file_chunks = self._chunk_code_file(content, rel_path)

                for chunk_text in file_chunks:
                    all_chunks.append(
                        {
                            "text": chunk_text,
                            "index": chunk_index,
                            "metadata": {
                                "file_path": rel_path,
                                "file_ext": Path(file_path).suffix,
                            },
                        }
                    )
                    chunk_index += 1

            except Exception as e:
                logger.warning("Failed to process file %s: %s", file_path, e)
                continue

        if not all_chunks:
            rag_item.status = "failed"
            await self.db.flush()
            return {
                "repo_name": repo_name,
                "files_indexed": 0,
                "chunks_created": 0,
            }

        # Generate embeddings in batch
        logger.info("Generating embeddings for %s chunks", len(all_chunks))
        chunk_texts = [c["text"] for c in all_chunks]
        embeddings = await embed_texts(chunk_texts, batch_size=32)

        # Create chunk records
        chunk_records = []
        for chunk_data, embedding in zip(all_chunks, embeddings, strict=False):
            chunk_record = RagChunk(
                user_id=self.user_id,
                rag_item_id=rag_item.id,
                content=chunk_data["text"],
                chunk_index=chunk_data["index"],
                embedding=embedding,
                chunk_metadata=chunk_data["metadata"],
            )
            chunk_records.append(chunk_record)

        self.db.add_all(chunk_records)

        # Update RAG item
        rag_item.chunk_count = len(chunk_records)
        rag_item.status = "indexed"
        await self.db.flush()

        logger.info("Indexed %s: %s files, %s chunks", repo_name, len(code_files), len(chunk_records))

        return {
            "repo_name": repo_name,
            "files_indexed": len(code_files),
            "chunks_created": len(chunk_records),
            "rag_item_id": rag_item.id,
        }

    async def search_code(
        self,
        repo_name: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search code in indexed repository using semantic search.

        Args:
            repo_name: Repository name
            query: Search query
            top_k: Number of results

        Returns:
            List of relevant code chunks
        """
        from sqlalchemy import text as sql_text

        from app.services.embedding_service import embed_text

        # Find RAG item for repository
        result = await self.db.execute(
            select(RagItem).where(
                RagItem.user_id == self.user_id,
                RagItem.item_metadata["repo_name"].astext == repo_name,
                RagItem.status == "indexed",
            )
        )
        rag_item = result.scalar_one_or_none()

        if not rag_item:
            return []

        # Generate query embedding
        query_embedding = await embed_text(query)

        # Semantic search
        search_query = sql_text("""
            SELECT
                rc.content,
                rc.chunk_metadata,
                1 - (rc.embedding <=> :query_embedding) as similarity_score
            FROM rag_chunks rc
            WHERE rc.rag_item_id = :rag_item_id
            ORDER BY rc.embedding <=> :query_embedding
            LIMIT :limit
        """)

        result = await self.db.execute(
            search_query,
            {
                "query_embedding": query_embedding,
                "rag_item_id": rag_item.id,
                "limit": top_k,
            },
        )

        rows = result.fetchall()

        return [
            {
                "content": row[0],
                "file_path": row[1].get("file_path"),
                "file_ext": row[1].get("file_ext"),
                "similarity_score": float(row[2]),
            }
            for row in rows
        ]

    async def read_file(self, repo_name: str, file_path: str) -> str:
        """
        Read file from repository.

        Args:
            repo_name: Repository name
            file_path: Relative file path

        Returns:
            File content
        """
        repo_path = self.sandbox.get_repos_path(repo_name)
        return await self.sandbox.read_file(file_path, base_dir=repo_path)

    async def write_file(
        self,
        repo_name: str,
        file_path: str,
        content: str,
    ) -> str:
        """
        Write file to repository.

        Args:
            repo_name: Repository name
            file_path: Relative file path
            content: File content

        Returns:
            Absolute path to written file
        """
        repo_path = self.sandbox.get_repos_path(repo_name)
        return await self.sandbox.write_file(file_path, content, base_dir=repo_path)

    async def create_commit(
        self,
        repo_name: str,
        message: str,
        files: list[str],
    ) -> dict[str, Any]:
        """
        Create git commit.

        Args:
            repo_name: Repository name
            message: Commit message
            files: List of file paths to commit

        Returns:
            Commit info
        """
        repo_path = self.sandbox.get_repos_path(repo_name)

        try:
            repo = Repo(repo_path)

            # Add files
            for file_path in files:
                repo.index.add([file_path])

            # Commit
            commit = repo.index.commit(message)

            logger.info("Created commit %s in %s", commit.hexsha[:8], repo_name)

            return {
                "commit_sha": commit.hexsha,
                "message": message,
                "files": files,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error("Failed to create commit: %s", e)
            raise GitHubError(f"Failed to create commit: {str(e)}") from e

    async def create_pull_request(
        self,
        repo_full_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
    ) -> dict[str, Any]:
        """
        Create pull request on GitHub.

        Args:
            repo_full_name: Repository full name (owner/repo)
            title: PR title
            body: PR description
            head_branch: Source branch
            base_branch: Target branch

        Returns:
            PR info
        """
        if not self.github:
            raise GitHubError("GitHub token not configured")

        try:
            repo = self.github.get_repo(repo_full_name)
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base_branch,
            )

            logger.info("Created PR #%s in %s", pr.number, repo_full_name)

            return {
                "pr_number": pr.number,
                "pr_url": pr.html_url,
                "title": title,
                "state": pr.state,
            }

        except Exception as e:
            logger.error("Failed to create PR: %s", e)
            raise GitHubError(f"Failed to create pull request: {str(e)}") from e

    def _get_directory_size(self, path: str) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for root, _dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                with contextlib.suppress(OSError):
                    total_size += os.path.getsize(file_path)
        return total_size

    def _chunk_code_file(self, content: str, file_path: str) -> list[str]:
        """
        Chunk code file by logical units (functions, classes).

        Simple heuristic: split by blank lines and group into chunks.
        """
        lines = content.split("\n")
        chunks = []
        current_chunk = []
        current_size = 0
        max_chunk_size = 1000  # characters

        for line in lines:
            current_chunk.append(line)
            current_size += len(line) + 1

            # Split on blank lines or when chunk is large enough
            if (not line.strip() and current_size > 200) or current_size > max_chunk_size:
                if current_chunk:
                    chunk_text = "\n".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                current_chunk = []
                current_size = 0

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks if chunks else [content]
