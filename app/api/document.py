from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pathlib import Path
import os
import shutil
from datetime import datetime

from app.db.postgres import get_db, DocumentRepository, Document
from app.db.qdrant import qdrant_manager
from app.services.parse import document_processor
from app.services.embedding import embedding_service
from app.core.settings import settings

router = APIRouter(prefix="/api/documents", tags=["documents"])


# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a document (PDF, MD, TXT)

    The service will:
    1. Validate and save the file
    2. Parse the text content
    3. Split into chunks
    4. Generate embeddings
    5. Store in Qdrant
    6. Save metadata in PostgreSQL

    Returns:
        Document metadata and processing status
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
        )

    # Check file size (this is approximate, actual size checked during read)
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
        )

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            if len(content) > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
                )
            buffer.write(content)

        file_size = os.path.getsize(file_path)
        file_type = file_ext.replace(".", "")

        # Create document record in PostgreSQL
        repo = DocumentRepository(db)
        document = await repo.create_document(
            filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_type
        )

        try:
            # Process document: parse, clean, and chunk
            full_text, chunks, preview = await document_processor.process_document(
                file_path=file_path,
                file_type=file_type
            )

            if not chunks:
                await repo.update_document_status(document.id, "failed")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to extract text from document"
                )

            # Generate embeddings for all chunks
            embeddings = await embedding_service.embed_chunks(chunks)

            # Store in Qdrant
            num_chunks = await qdrant_manager.add_documents(
                document_id=document.id,
                chunks=chunks,
                embeddings=embeddings,
                metadata={
                    "filename": file.filename,
                    "file_type": file_type,
                    "upload_date": document.upload_date.isoformat()
                }
            )

            # Update document status and chunk count
            document = await repo.update_document_status(
                document_id=document.id,
                status="completed",
                num_chunks=num_chunks
            )

            # Update preview
            document.content_preview = preview
            await db.commit()
            await db.refresh(document)

            return {
                "status": "success",
                "message": "Document uploaded and processed successfully",
                "document": document.to_dict(),
                "chunks_created": num_chunks
            }

        except Exception as e:
            # Update status to failed
            await repo.update_document_status(document.id, "failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing document: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


@router.get("/", response_model=List[dict])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """
    List all documents with pagination

    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return

    Returns:
        List of document metadata
    """
    repo = DocumentRepository(db)
    documents = await repo.get_all_documents(skip=skip, limit=limit)
    return [doc.to_dict() for doc in documents]


@router.get("/{document_id}")
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get document details by ID

    Args:
        document_id: ID of the document

    Returns:
        Document metadata
    """
    repo = DocumentRepository(db)
    document = await repo.get_document(document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return document.to_dict()


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get all chunks for a specific document

    Args:
        document_id: ID of the document

    Returns:
        List of chunks with their text and metadata
    """
    repo = DocumentRepository(db)
    document = await repo.get_document(document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    chunks = await qdrant_manager.get_document_chunks(document_id)

    return {
        "document_id": document_id,
        "filename": document.filename,
        "chunks": chunks,
        "total_chunks": len(chunks)
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a document and all its chunks

    Args:
        document_id: ID of the document to delete

    Returns:
        Deletion status
    """
    repo = DocumentRepository(db)
    document = await repo.get_document(document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    try:
        # Delete from Qdrant
        await qdrant_manager.delete_document(document_id)

        # Delete file from disk
        if os.path.exists(document.file_path):
            os.remove(document.file_path)

        # Delete from PostgreSQL
        await repo.delete_document(document_id)

        return {
            "status": "success",
            "message": f"Document {document_id} deleted successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get("/search/query")
async def search_documents(
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=100),
    document_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Search for similar document chunks using semantic search

    Args:
        query: Search query text
        limit: Maximum number of results
        document_id: Optional filter by specific document

    Returns:
        List of similar chunks with relevance scores
    """
    try:
        # Generate query embedding
        query_embedding = await embedding_service.embed_query(query)

        # Search in Qdrant
        results = await qdrant_manager.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            document_id=document_id,
            score_threshold=0.3
        )

        # Enrich results with document metadata
        enriched_results = []
        repo = DocumentRepository(db)

        for result in results:
            doc_id = result["document_id"]
            document = await repo.get_document(doc_id)

            enriched_results.append({
                "score": result["score"],
                "text": result["text"],
                "chunk_index": result["chunk_index"],
                "document": {
                    "id": doc_id,
                    "filename": document.filename if document else "Unknown",
                    "file_type": document.file_type if document else "Unknown"
                }
            })

        return {
            "query": query,
            "results": enriched_results,
            "total_results": len(enriched_results)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching documents: {str(e)}"
        )


@router.get("/search/by-name")
async def search_by_filename(
    query: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db)
):
    """
    Search documents by filename

    Args:
        query: Filename search query

    Returns:
        List of matching documents
    """
    repo = DocumentRepository(db)
    documents = await repo.search_documents(query)
    return [doc.to_dict() for doc in documents]


@router.get("/stats/overview")
async def get_statistics(db: AsyncSession = Depends(get_db)):
    """
    Get overall statistics about documents

    Returns:
        Statistics about uploaded documents
    """
    repo = DocumentRepository(db)
    documents = await repo.get_all_documents(skip=0, limit=10000)

    total_documents = len(documents)
    total_size = sum(doc.file_size for doc in documents)
    total_chunks = sum(doc.num_chunks for doc in documents)

    by_type = {}
    by_status = {}

    for doc in documents:
        by_type[doc.file_type] = by_type.get(doc.file_type, 0) + 1
        by_status[doc.status] = by_status.get(doc.status, 0) + 1

    # Get Qdrant collection info
    collection_info = await qdrant_manager.get_collection_info()

    return {
        "total_documents": total_documents,
        "total_size_bytes": total_size,
        "total_chunks": total_chunks,
        "by_file_type": by_type,
        "by_status": by_status,
        "vector_store": collection_info
    }
