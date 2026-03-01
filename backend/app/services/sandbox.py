"""
Secure Sandbox — izolowane środowisko dla operacji filesystem i git.

Bezpieczeństwo:
- Ścisła walidacja ścieżek (path traversal protection)
- Whitelist dozwolonych operacji
- Limity rozmiaru plików
- Timeout dla operacji
- Izolacja per-user (każdy user ma swój katalog)
- Automatyczne czyszczenie starych plików

Architektura:
    /tmp/nexus_sandbox/
        /{user_id}/
            /repos/          # Sklonowane repozytoria
            /workspace/      # Obszar roboczy
            /temp/           # Pliki tymczasowe
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.core.exceptions import SandboxError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class Sandbox:
    """
    Secure sandbox for filesystem and git operations.

    Provides isolated, per-user environment with security controls.
    """

    # Base sandbox directory
    BASE_DIR = "/tmp/nexus_sandbox"

    # Subdirectories
    REPOS_DIR = "repos"
    WORKSPACE_DIR = "workspace"
    TEMP_DIR = "temp"

    # Security limits
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_FILES_PER_USER = 1000
    MAX_TOTAL_SIZE_PER_USER = 100 * 1024 * 1024  # 100MB
    CLEANUP_AGE_DAYS = 7  # Auto-delete files older than 7 days

    # Allowed file extensions for write operations
    ALLOWED_EXTENSIONS = {
        ".txt",
        ".md",
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".c",
        ".go",
        ".rs",
        ".html",
        ".css",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".sh",
        ".sql",
        ".gitignore",
        ".env.example",
        ".dockerignore",
    }

    # Forbidden path components
    FORBIDDEN_PATHS = {"..", "~", "/etc", "/sys", "/proc", "/root", "/home"}

    def __init__(self, user_id: int) -> None:
        """
        Initialize sandbox for a specific user.

        Args:
            user_id: User's Telegram ID
        """
        self.user_id = user_id
        self.user_dir = os.path.join(self.BASE_DIR, str(user_id))
        self.repos_dir = os.path.join(self.user_dir, self.REPOS_DIR)
        self.workspace_dir = os.path.join(self.user_dir, self.WORKSPACE_DIR)
        self.temp_dir = os.path.join(self.user_dir, self.TEMP_DIR)

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create sandbox directories if they don't exist."""
        for directory in [self.user_dir, self.repos_dir, self.workspace_dir, self.temp_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _validate_path(self, path: str, base_dir: str | None = None) -> str:
        """
        Validate and normalize path to prevent path traversal attacks.

        Args:
            path: Path to validate
            base_dir: Base directory to restrict to (defaults to user_dir)

        Returns:
            Absolute, validated path

        Raises:
            SandboxError: If path is invalid or outside sandbox
        """
        base_dir = base_dir or self.user_dir

        # Check for forbidden components
        for forbidden in self.FORBIDDEN_PATHS:
            if forbidden in path:
                raise SandboxError(
                    f"Forbidden path component: {forbidden}",
                    {"path": path, "forbidden": forbidden},
                )

        # Resolve to absolute path
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)

        # Normalize and resolve symlinks
        abs_path = os.path.abspath(os.path.realpath(path))
        abs_base = os.path.abspath(os.path.realpath(base_dir))

        # Ensure path is within base directory
        if not abs_path.startswith(abs_base):
            raise SandboxError(
                "Path outside sandbox",
                {"path": abs_path, "base": abs_base},
            )

        return abs_path

    def _check_file_size(self, file_path: str) -> None:
        """
        Check if file size is within limits.

        Args:
            file_path: Path to file

        Raises:
            SandboxError: If file too large
        """
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size > self.MAX_FILE_SIZE:
                raise SandboxError(
                    f"File too large: {size} bytes (max {self.MAX_FILE_SIZE})",
                    {"file": file_path, "size": size},
                )

    def _check_user_quota(self) -> None:
        """
        Check if user is within storage quota.

        Raises:
            SandboxError: If quota exceeded
        """
        total_size = 0
        file_count = 0

        for root, _dirs, files in os.walk(self.user_dir):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                file_count += 1

        if file_count > self.MAX_FILES_PER_USER:
            raise SandboxError(
                f"Too many files: {file_count} (max {self.MAX_FILES_PER_USER})",
                {"count": file_count},
            )

        if total_size > self.MAX_TOTAL_SIZE_PER_USER:
            raise SandboxError(
                f"Storage quota exceeded: {total_size} bytes (max {self.MAX_TOTAL_SIZE_PER_USER})",
                {"size": total_size},
            )

    async def read_file(self, path: str, base_dir: str | None = None) -> str:
        """
        Read file content from sandbox.

        Args:
            path: Relative or absolute path within sandbox
            base_dir: Base directory (defaults to workspace)

        Returns:
            File content as string

        Raises:
            SandboxError: If path invalid or file not found
        """
        base_dir = base_dir or self.workspace_dir
        abs_path = self._validate_path(path, base_dir)

        if not os.path.exists(abs_path):
            raise SandboxError(f"File not found: {path}", {"path": abs_path})

        if not os.path.isfile(abs_path):
            raise SandboxError(f"Not a file: {path}", {"path": abs_path})

        self._check_file_size(abs_path)

        try:
            with open(abs_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            logger.info("Read file: %s (%s chars)", abs_path, len(content))
            return content
        except Exception as e:
            raise SandboxError(f"Failed to read file: {str(e)}", {"path": abs_path}) from e

    async def write_file(
        self,
        path: str,
        content: str,
        base_dir: str | None = None,
    ) -> str:
        """
        Write file to sandbox.

        Args:
            path: Relative or absolute path within sandbox
            content: File content
            base_dir: Base directory (defaults to workspace)

        Returns:
            Absolute path to written file

        Raises:
            SandboxError: If path invalid or write fails
        """
        base_dir = base_dir or self.workspace_dir
        abs_path = self._validate_path(path, base_dir)

        # Check file extension
        file_ext = Path(abs_path).suffix.lower()
        if file_ext and file_ext not in self.ALLOWED_EXTENSIONS:
            raise SandboxError(
                f"File extension not allowed: {file_ext}",
                {"path": abs_path, "extension": file_ext},
            )

        # Check content size
        content_size = len(content.encode("utf-8"))
        if content_size > self.MAX_FILE_SIZE:
            raise SandboxError(
                f"Content too large: {content_size} bytes",
                {"size": content_size},
            )

        # Check user quota
        self._check_user_quota()

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("Wrote file: %s (%s bytes)", abs_path, content_size)
            return abs_path
        except Exception as e:
            raise SandboxError(f"Failed to write file: {str(e)}", {"path": abs_path}) from e

    async def list_files(
        self,
        path: str = ".",
        base_dir: str | None = None,
        recursive: bool = False,
    ) -> list[dict[str, Any]]:
        """
        List files in directory.

        Args:
            path: Directory path (defaults to workspace root)
            base_dir: Base directory (defaults to workspace)
            recursive: List recursively

        Returns:
            List of file info dicts

        Raises:
            SandboxError: If path invalid
        """
        base_dir = base_dir or self.workspace_dir
        abs_path = self._validate_path(path, base_dir)

        if not os.path.exists(abs_path):
            raise SandboxError(f"Directory not found: {path}", {"path": abs_path})

        if not os.path.isdir(abs_path):
            raise SandboxError(f"Not a directory: {path}", {"path": abs_path})

        files = []

        if recursive:
            for root, _dirs, filenames in os.walk(abs_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, abs_path)
                    files.append(self._get_file_info(file_path, rel_path))
        else:
            for item in os.listdir(abs_path):
                item_path = os.path.join(abs_path, item)
                files.append(self._get_file_info(item_path, item))

        return files

    def _get_file_info(self, abs_path: str, rel_path: str) -> dict[str, Any]:
        """Get file information."""
        stat = os.stat(abs_path)
        return {
            "name": rel_path,
            "path": abs_path,
            "type": "file" if os.path.isfile(abs_path) else "directory",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        }

    async def delete_file(self, path: str, base_dir: str | None = None) -> bool:
        """
        Delete file from sandbox.

        Args:
            path: File path
            base_dir: Base directory (defaults to workspace)

        Returns:
            True if deleted

        Raises:
            SandboxError: If path invalid
        """
        base_dir = base_dir or self.workspace_dir
        abs_path = self._validate_path(path, base_dir)

        if not os.path.exists(abs_path):
            return False

        try:
            if os.path.isfile(abs_path):
                os.remove(abs_path)
            elif os.path.isdir(abs_path):
                shutil.rmtree(abs_path)
            logger.info("Deleted: %s", abs_path)
            return True
        except Exception as e:
            raise SandboxError(f"Failed to delete: {str(e)}", {"path": abs_path}) from e

    async def cleanup_old_files(self) -> int:
        """
        Clean up files older than CLEANUP_AGE_DAYS.

        Returns:
            Number of files deleted
        """
        cutoff_time = datetime.now(UTC) - timedelta(days=self.CLEANUP_AGE_DAYS)
        deleted_count = 0

        for root, _dirs, files in os.walk(self.user_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path), tz=UTC)
                    if mtime < cutoff_time:
                        os.remove(file_path)
                        deleted_count += 1
                except Exception as e:
                    logger.warning("Failed to delete old file %s: %s", file_path, e)

        if deleted_count > 0:
            logger.info("Cleaned up %s old files for user %s", deleted_count, self.user_id)

        return deleted_count

    def get_workspace_path(self, relative_path: str = "") -> str:
        """
        Get absolute path in workspace.

        Args:
            relative_path: Relative path within workspace

        Returns:
            Absolute path
        """
        return self._validate_path(relative_path, self.workspace_dir)

    def get_repos_path(self, relative_path: str = "") -> str:
        """
        Get absolute path in repos directory.

        Args:
            relative_path: Relative path within repos

        Returns:
            Absolute path
        """
        return self._validate_path(relative_path, self.repos_dir)
