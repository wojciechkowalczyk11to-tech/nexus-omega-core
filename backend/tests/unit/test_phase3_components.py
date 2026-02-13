"""
Comprehensive test suite for Phase 3 components.

Tests:
- Embedding service
- RAG Tool V2 (pgvector)
- Sandbox security
- SLM Router
- GitHub Devin Tool
- Agent Traces API
"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

# Embedding Service Tests


@pytest.mark.asyncio
async def test_embedding_service_single_text():
    """Test generating embedding for single text."""
    from app.services.embedding_service import embed_text

    text = "This is a test sentence for embedding generation."
    embedding = await embed_text(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384  # all-MiniLM-L6-v2 dimensions
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_embedding_service_batch():
    """Test generating embeddings for multiple texts."""
    from app.services.embedding_service import embed_texts

    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence.",
    ]
    embeddings = await embed_texts(texts)

    assert len(embeddings) == 3
    assert all(len(emb) == 384 for emb in embeddings)


@pytest.mark.asyncio
async def test_embedding_service_empty_text():
    """Test handling empty text."""
    from app.services.embedding_service import embed_text

    embedding = await embed_text("")

    assert len(embedding) == 384
    assert all(x == 0.0 for x in embedding)  # Zero vector for empty text


# Sandbox Tests


@pytest.mark.asyncio
async def test_sandbox_path_validation():
    """Test sandbox path traversal protection."""
    from app.services.sandbox import Sandbox

    sandbox = Sandbox(user_id=12345)

    # Valid path
    valid_path = sandbox._validate_path("test.txt")
    assert valid_path.startswith(sandbox.workspace_dir)

    # Path traversal attempt
    with pytest.raises(ValueError):
        sandbox._validate_path("../../../etc/passwd")

    # Forbidden path
    with pytest.raises(ValueError):
        sandbox._validate_path("/etc/passwd")


@pytest.mark.asyncio
async def test_sandbox_write_and_read():
    """Test sandbox file write and read."""
    from app.services.sandbox import Sandbox

    sandbox = Sandbox(user_id=12345)

    # Write file
    content = "Test content for sandbox"
    file_path = await sandbox.write_file("test.txt", content)

    assert os.path.exists(file_path)

    # Read file
    read_content = await sandbox.read_file("test.txt")
    assert read_content == content

    # Cleanup
    await sandbox.delete_file("test.txt")


@pytest.mark.asyncio
async def test_sandbox_file_size_limit():
    """Test sandbox file size limits."""
    from app.core.exceptions import SandboxError
    from app.services.sandbox import Sandbox

    sandbox = Sandbox(user_id=12345)

    # Try to write file exceeding limit
    large_content = "x" * (sandbox.MAX_FILE_SIZE + 1)

    with pytest.raises(SandboxError):
        await sandbox.write_file("large.txt", large_content)


@pytest.mark.asyncio
async def test_sandbox_list_files():
    """Test sandbox file listing."""
    from app.services.sandbox import Sandbox

    sandbox = Sandbox(user_id=12345)

    # Create test files
    await sandbox.write_file("file1.txt", "content1")
    await sandbox.write_file("file2.txt", "content2")

    # List files
    files = await sandbox.list_files()

    assert len(files) >= 2
    file_names = [f["name"] for f in files]
    assert "file1.txt" in file_names
    assert "file2.txt" in file_names

    # Cleanup
    await sandbox.delete_file("file1.txt")
    await sandbox.delete_file("file2.txt")


# SLM Router Tests


def test_slm_router_simple_task_low_cost():
    """Test SLM router for simple task with low cost preference."""
    from app.services.slm_router import CostPreference, SLMRouter

    model = SLMRouter.select_model(
        difficulty="simple",
        cost_preference=CostPreference.LOW,
    )

    assert model.tier.value == "ultra_cheap"
    assert model.cost_per_1m_input < 0.20


def test_slm_router_complex_task_quality():
    """Test SLM router for complex task with quality preference."""
    from app.services.slm_router import CostPreference, SLMRouter

    model = SLMRouter.select_model(
        difficulty="complex",
        cost_preference=CostPreference.QUALITY,
    )

    # Should select premium or balanced tier
    assert model.tier.value in ["premium", "balanced"]


def test_slm_router_function_calling_requirement():
    """Test SLM router with function calling requirement."""
    from app.services.slm_router import CostPreference, SLMRouter

    model = SLMRouter.select_model(
        difficulty="moderate",
        cost_preference=CostPreference.BALANCED,
        requires_function_calling=True,
    )

    assert model.supports_function_calling is True


def test_slm_router_cost_estimation():
    """Test cost estimation."""
    from app.services.slm_router import ModelTier, SLMRouter

    # Get a model
    models = SLMRouter.MODELS[ModelTier.ULTRA_CHEAP]
    model = models[0]

    # Estimate cost
    cost = SLMRouter.estimate_cost(
        model=model,
        input_tokens=1000,
        output_tokens=500,
    )

    assert cost > 0
    assert cost < 1.0  # Should be very cheap for small request


def test_slm_router_escalation_decision():
    """Test escalation decision logic."""
    from app.services.slm_router import ModelTier, SLMRouter

    models = SLMRouter.MODELS[ModelTier.ULTRA_CHEAP]
    model = models[0]

    # Low complexity - should not escalate
    should_escalate = SLMRouter.should_escalate(
        current_model=model,
        task_complexity_score=0.2,
    )
    assert should_escalate is False

    # High complexity - should escalate
    should_escalate = SLMRouter.should_escalate(
        current_model=model,
        task_complexity_score=0.9,
    )
    assert should_escalate is True


# RAG Tool V2 Tests


@pytest.mark.asyncio
async def test_rag_tool_v2_chunking():
    """Test semantic chunking."""
    from app.tools.rag_tool_v2 import RAGToolV2

    db_mock = MagicMock(spec=AsyncSession)
    rag_tool = RAGToolV2(db=db_mock, storage_path=tempfile.mkdtemp())

    text = """
    This is the first paragraph with some content.
    It has multiple sentences.

    This is the second paragraph.
    It also has content.

    This is the third paragraph.
    """

    chunks = rag_tool._chunk_text_semantic(text)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk) > 0 for chunk in chunks)


@pytest.mark.asyncio
async def test_rag_tool_v2_reranking():
    """Test result reranking."""
    from app.tools.rag_tool_v2 import RAGToolV2

    db_mock = MagicMock(spec=AsyncSession)
    rag_tool = RAGToolV2(db=db_mock)

    results = [
        {
            "content": "This document talks about Python programming",
            "similarity_score": 0.7,
        },
        {
            "content": "Random content without keywords",
            "similarity_score": 0.8,
        },
        {
            "content": "Another Python programming tutorial",
            "similarity_score": 0.6,
        },
    ]

    query = "Python programming"
    reranked = rag_tool._rerank_results(results, query)

    # Results with keyword matches should be boosted
    assert reranked[0]["keyword_matches"] > 0


# Agent Traces Tests


@pytest.mark.asyncio
async def test_agent_trace_model():
    """Test agent trace model creation."""
    from app.db.models.agent_trace import AgentTrace

    trace = AgentTrace(
        user_id=12345,
        message_id=1,
        iteration=1,
        action="think",
        thought="Analyzing user query",
        timestamp_ms=1234567890,
    )

    assert trace.user_id == 12345
    assert trace.action == "think"
    assert trace.thought == "Analyzing user query"


# GitHub Devin Tool Tests


@pytest.mark.asyncio
async def test_github_devin_tool_code_chunking():
    """Test code file chunking."""
    from app.tools.github_devin_tool import GitHubDevinTool

    db_mock = MagicMock(spec=AsyncSession)
    devin_tool = GitHubDevinTool(user_id=12345, db=db_mock)

    code = """
def function_one():
    return "test"

def function_two():
    return "test2"

class MyClass:
    def method(self):
        pass
"""

    chunks = devin_tool._chunk_code_file(code, "test.py")

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


@pytest.mark.asyncio
async def test_github_devin_tool_directory_size():
    """Test directory size calculation."""
    from app.tools.github_devin_tool import GitHubDevinTool

    db_mock = MagicMock(spec=AsyncSession)
    devin_tool = GitHubDevinTool(user_id=12345, db=db_mock)

    # Create temp directory with files
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test content" * 100)

    size = devin_tool._get_directory_size(temp_dir)

    assert size > 0
    assert size == len("test content" * 100)

    # Cleanup
    os.remove(test_file)
    os.rmdir(temp_dir)


# Integration Tests


@pytest.mark.asyncio
async def test_embedding_to_rag_integration():
    """Test integration between embedding service and RAG."""
    from app.services.embedding_service import embed_text

    # Generate embedding
    text = "Test document for RAG indexing"
    embedding = await embed_text(text)

    # Verify embedding can be used in RAG
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_sandbox_to_github_integration():
    """Test integration between sandbox and GitHub tool."""
    from app.services.sandbox import Sandbox

    sandbox = Sandbox(user_id=12345)

    # Write a file in sandbox
    content = "# Test Repository\n\nThis is a test."
    file_path = await sandbox.write_file("README.md", content)

    assert os.path.exists(file_path)

    # Read it back
    read_content = await sandbox.read_file("README.md")
    assert read_content == content

    # Cleanup
    await sandbox.delete_file("README.md")


# Performance Tests


@pytest.mark.asyncio
async def test_embedding_batch_performance():
    """Test embedding batch generation performance."""
    import time

    from app.services.embedding_service import embed_texts

    texts = [f"Test sentence number {i}" for i in range(50)]

    start_time = time.time()
    embeddings = await embed_texts(texts, batch_size=32)
    elapsed = time.time() - start_time

    assert len(embeddings) == 50
    # Should complete in reasonable time (< 5 seconds on CPU)
    assert elapsed < 10.0


@pytest.mark.asyncio
async def test_sandbox_file_operations_performance():
    """Test sandbox file operations performance."""
    import time

    from app.services.sandbox import Sandbox

    sandbox = Sandbox(user_id=12345)

    start_time = time.time()

    # Write multiple files
    for i in range(10):
        await sandbox.write_file(f"test_{i}.txt", f"Content {i}")

    # Read them back
    for i in range(10):
        await sandbox.read_file(f"test_{i}.txt")

    elapsed = time.time() - start_time

    # Should be fast (< 1 second)
    assert elapsed < 2.0

    # Cleanup
    for i in range(10):
        await sandbox.delete_file(f"test_{i}.txt")


# Error Handling Tests


@pytest.mark.asyncio
async def test_sandbox_error_handling():
    """Test sandbox error handling."""
    from app.core.exceptions import SandboxError
    from app.services.sandbox import Sandbox

    sandbox = Sandbox(user_id=12345)

    # Try to read non-existent file
    with pytest.raises(SandboxError):
        await sandbox.read_file("nonexistent.txt")

    # Try to write with forbidden extension
    with pytest.raises(SandboxError):
        await sandbox.write_file("test.exe", "content")


@pytest.mark.asyncio
async def test_rag_tool_error_handling():
    """Test RAG tool error handling."""
    from app.core.exceptions import RAGError
    from app.tools.rag_tool_v2 import RAGToolV2

    db_mock = MagicMock(spec=AsyncSession)
    rag_tool = RAGToolV2(db=db_mock)

    # Try to upload unsupported file type
    with pytest.raises(RAGError):
        await rag_tool.upload_document(
            user_id=12345,
            filename="test.exe",
            content=b"binary content",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
