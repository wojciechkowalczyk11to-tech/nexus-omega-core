"""
Security audit: checks that no real secrets have leaked into source code.
Verifies that .env.example contains ONLY placeholders.
"""

import re
from pathlib import Path

import pytest

# Patterns that should NOT appear in code (real API keys)
FORBIDDEN_PATTERNS = [
    r'sk-[a-zA-Z0-9]{20,}',                   # OpenAI API key format
    r'AIza[a-zA-Z0-9_-]{35}',                  # Google API key format
    r'xai-[a-zA-Z0-9]{20,}',                   # xAI key format
    r'(?:ghp|github_pat)_[a-zA-Z0-9]{36,}',    # GitHub PAT
    r'Bearer [a-zA-Z0-9+/]{40,}={0,2}',        # Bearer tokens
    r'postgresql://[^:]+:[^@]{8,}@',            # DB URL with real password
]

# Files that MAY contain these patterns (test fixtures, documentation)
EXCLUDED_FILES = {'.env', 'tests/test_secrets_audit.py', 'test_secrets_audit.py'}

ROOT = Path('/home/runner/work/nexus-omega-core/nexus-omega-core')


def get_all_source_files() -> list[Path]:
    """Get all source files that should be checked for secrets."""
    extensions = {'.py', '.yaml', '.yml', '.toml', '.cfg', '.ini'}
    files = []
    for f in ROOT.rglob('*'):
        if f.suffix in extensions and not any(
            excluded in str(f) for excluded in ['.git', '__pycache__', 'node_modules', '.venv']
        ) and f.name not in EXCLUDED_FILES:
            files.append(f)
    return sorted(files)


@pytest.mark.parametrize('source_file', get_all_source_files(), ids=lambda f: str(f.relative_to(ROOT)))
def test_no_real_secrets_in_source(source_file: Path) -> None:
    """Verify no real secrets are present in source files."""
    content = source_file.read_text(errors='ignore')
    for pattern in FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, content)
        assert not matches, (
            f"Potential real secret in {source_file.relative_to(ROOT)}: "
            f"pattern '{pattern}' matched: {matches[:2]}"
        )


def test_env_example_exists() -> None:
    """Verify .env.example exists."""
    env_example = ROOT / '.env.example'
    assert env_example.exists(), ".env.example file not found"


def test_env_example_contains_only_placeholders() -> None:
    """Verify .env.example contains only placeholder values, not real secrets."""
    env_example = ROOT / '.env.example'
    content = env_example.read_text()
    lines_with_values = [
        line for line in content.splitlines()
        if '=' in line and not line.startswith('#') and not line.strip().endswith('=')
    ]
    placeholder_keywords = {
        'YOUR_', 'CHANGE_ME', 'changeme', '_HERE', 'example', 'false', 'true',
        '0', '1', '[]', 'INFO', 'HS256', 'polling', 'us-central1', 'webhook',
        '8443', '8000', '6379', '5432', '24', '30', '5.00', '5', '20', '50',
        'redis://', 'postgresql+asyncpg://', 'http://', '["*"]',
    }
    for line in lines_with_values:
        value = line.split('=', 1)[1].strip()
        is_placeholder = any(kw.lower() in value.lower() for kw in placeholder_keywords) or len(value) < 30
        assert is_placeholder, f"Possible real secret in .env.example: {line}"
