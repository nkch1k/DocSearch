from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import time
import logging

from app.db.postgres import get_db, DocumentRepository
from app.services.llm import llm_service
from app.services.retrieval import retrieval_service
from app.models.answer import (
    QuestionRequest,
    AnswerResponse,
    DocumentSummaryRequest,
    DocumentSummaryResponse,
    ChunkInfo,
    SourceInfo,
    TokenUsage,
    HealthCheckResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/answer", tags=["answer"])


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Ask a question and get an AI-generated answer based on document context

    This endpoint implements RAG (Retrieval-Augmented Generation):
    1. Retrieves relevant context from vector database (Qdrant)
    2. Falls back to PostgreSQL if vector search fails
    3. Generates answer using LLM with retrieved context
    4. Returns answer with sources and metadata

    Args:
        request: Question request with parameters
        db: Database session

    Returns:
        Answer with sources and metadata
    """
    start_time = time.time()

    try:
        # Step 1: Retrieve relevant context
        logger.info(f"Retrieving context for question: {request.question[:50]}...")

        context_chunks = await retrieval_service.retrieve_context(
            question=request.question,
            db=db,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            document_id=request.document_id,
            use_postgres_fallback=request.use_postgres_fallback
        )

        if not context_chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found for your question. Please try rephrasing or upload relevant documents."
            )

        # Determine retrieval method
        retrieval_method = "vector_search"
        if context_chunks and context_chunks[0].get("score", 1.0) < 0.5:
            # Low scores might indicate fallback was used
            retrieval_method = "hybrid"

        logger.info(f"Retrieved {len(context_chunks)} relevant chunks")

        # Step 2: Generate answer using LLM
        logger.info("Generating answer with LLM...")

        llm_response = await llm_service.generate_answer(
            question=request.question,
            context_chunks=context_chunks,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # Step 3: Build response
        processing_time_ms = (time.time() - start_time) * 1000

        response = AnswerResponse(
            question=request.question,
            answer=llm_response["answer"],
            sources=[
                SourceInfo(**source) for source in llm_response["sources"]
            ],
            chunks_used=[
                ChunkInfo(**chunk) for chunk in context_chunks
            ],
            model=llm_response["model"],
            tokens_used=TokenUsage(**llm_response["tokens_used"]),
            retrieval_method=retrieval_method,
            processing_time_ms=processing_time_ms
        )

        logger.info(f"Answer generated successfully in {processing_time_ms:.2f}ms")
        return response

    except HTTPException:
        raise
    except ValueError as e:
        # LLM service not configured
        logger.error(f"Configuration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {str(e)}"
        )


@router.post("/summarize", response_model=DocumentSummaryResponse)
async def summarize_document(
    request: DocumentSummaryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a summary of a specific document

    Args:
        request: Summary request
        db: Database session

    Returns:
        Document summary
    """
    try:
        # Get document context
        document_context = await retrieval_service.get_full_document_context(
            document_id=request.document_id,
            db=db
        )

        if not document_context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {request.document_id} not found"
            )

        # Generate summary
        summary = await llm_service.generate_summary(
            text=document_context["full_text"],
            max_length=request.max_length
        )

        # Note: For simplicity, we're not tracking tokens in summary
        # You could extend this to get actual token usage
        response = DocumentSummaryResponse(
            document_id=request.document_id,
            filename=document_context["filename"],
            summary=summary,
            model=llm_service.model,
            tokens_used=TokenUsage(prompt=0, completion=0, total=0)  # Placeholder
        )

        return response

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating summary: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Check health of RAG system components

    Returns:
        Health status of all components
    """
    # Check LLM
    llm_available = llm_service.client is not None

    # Check vector database
    vector_db_available = True
    try:
        from app.db.qdrant import qdrant_manager
        await qdrant_manager.get_collection_info()
    except Exception as e:
        logger.warning(f"Vector DB health check failed: {e}")
        vector_db_available = False

    # Check PostgreSQL
    postgres_available = True
    try:
        repo = DocumentRepository(db)
        await repo.get_document_stats()
    except Exception as e:
        logger.warning(f"PostgreSQL health check failed: {e}")
        postgres_available = False

    # Overall status
    if llm_available and (vector_db_available or postgres_available):
        overall_status = "healthy"
    elif postgres_available:
        overall_status = "degraded"  # Can work with fallback
    else:
        overall_status = "unhealthy"

    return HealthCheckResponse(
        status=overall_status,
        llm_available=llm_available,
        vector_db_available=vector_db_available,
        postgres_available=postgres_available
    )


@router.get("/context/{document_id}")
async def get_document_context(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get full context of a document (for debugging/inspection)

    Args:
        document_id: Document ID
        db: Database session

    Returns:
        Full document context with all chunks
    """
    context = await retrieval_service.get_full_document_context(
        document_id=document_id,
        db=db
    )

    if not context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )

    return context
