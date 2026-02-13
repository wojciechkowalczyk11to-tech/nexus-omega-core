"""
RAG API routes for document management.
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.exceptions import RAGError
from app.db.models.user import User
from app.tools.rag_tool import RAGTool

router = APIRouter()


class RagItemResponse(BaseModel):
    """RAG item response schema."""

    id: int
    filename: str
    source_type: str
    source_url: str | None
    chunk_count: int
    status: str
    created_at: str


class RagListResponse(BaseModel):
    """RAG list response schema."""

    items: list[RagItemResponse]
    total: int


class RagUploadResponse(BaseModel):
    """RAG upload response schema."""

    item_id: int
    filename: str
    chunk_count: int
    message: str


class RagDeleteResponse(BaseModel):
    """RAG delete response schema."""

    success: bool
    message: str


@router.post("/upload", response_model=RagUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RagUploadResponse:
    """
    Upload a document for RAG.

    Supported formats: .txt, .md, .pdf, .docx, .html, .json
    """
    # Check if user has FULL_ACCESS
    if current_user.role not in ("FULL_ACCESS", "ADMIN"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="RAG upload wymaga roli FULL_ACCESS. Użyj /subscribe",
        )

    rag_tool = RAGTool(db)

    try:
        # Read file content
        content = await file.read()

        # Upload and process
        rag_item = await rag_tool.upload_document(
            user_id=current_user.telegram_id,
            filename=file.filename or "untitled",
            content=content,
            scope="user",
        )

        await db.commit()

        return RagUploadResponse(
            item_id=rag_item.id,
            filename=rag_item.filename,
            chunk_count=rag_item.chunk_count,
            message=f"Dokument '{rag_item.filename}' został przetworzony ({rag_item.chunk_count} fragmentów)",
        )

    except RAGError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        ) from e


@router.get("/list", response_model=RagListResponse)
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RagListResponse:
    """
    List user's RAG documents.
    """
    rag_tool = RAGTool(db)

    items = await rag_tool.list_documents(current_user.telegram_id)

    return RagListResponse(
        items=[
            RagItemResponse(
                id=item.id,
                filename=item.filename,
                source_type=item.source_type,
                source_url=item.source_url,
                chunk_count=item.chunk_count,
                status=item.status,
                created_at=item.created_at.isoformat(),
            )
            for item in items
        ],
        total=len(items),
    )


@router.delete("/{item_id}", response_model=RagDeleteResponse)
async def delete_document(
    item_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RagDeleteResponse:
    """
    Delete a RAG document.
    """
    rag_tool = RAGTool(db)

    success = await rag_tool.delete_document(current_user.telegram_id, item_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dokument nie istnieje",
        )

    await db.commit()

    return RagDeleteResponse(
        success=True,
        message="Dokument usunięty pomyślnie",
    )
