from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class QuestionRequest(BaseModel):
    """Request model for asking a question"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's question"
    )
    document_id: Optional[int] = Field(
        None,
        description="Optional: Limit search to specific document"
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve"
    )
    score_threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for chunks"
    )
    use_postgres_fallback: bool = Field(
        True,
        description="Use PostgreSQL as fallback if vector search fails"
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=100,
        le=4000,
        description="Maximum tokens in LLM response"
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation"
    )


class SourceInfo(BaseModel):
    """Information about a source document"""

    document_id: int = Field(..., description="Document ID")
    filename: str = Field(..., description="Document filename")
    file_type: str = Field(..., description="Document file type")
    relevance_score: float = Field(..., description="Relevance score")


class ChunkInfo(BaseModel):
    """Information about a retrieved chunk"""

    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Similarity score")
    chunk_index: int = Field(..., description="Chunk index in document")
    document: Dict = Field(..., description="Document metadata")


class TokenUsage(BaseModel):
    """Token usage information"""

    prompt: int = Field(..., description="Prompt tokens used")
    completion: int = Field(..., description="Completion tokens used")
    total: int = Field(..., description="Total tokens used")


class AnswerResponse(BaseModel):
    """Response model for question answers"""

    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(
        default_factory=list,
        description="Source documents used"
    )
    chunks_used: List[ChunkInfo] = Field(
        default_factory=list,
        description="Retrieved chunks used for context"
    )
    model: str = Field(..., description="LLM model used")
    tokens_used: TokenUsage = Field(..., description="Token usage statistics")
    retrieval_method: str = Field(
        ...,
        description="Method used for retrieval (vector/postgres_fallback)"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Total processing time in milliseconds"
    )


class DocumentSummaryRequest(BaseModel):
    """Request model for document summary"""

    document_id: int = Field(..., description="Document ID to summarize")
    max_length: int = Field(
        200,
        ge=50,
        le=1000,
        description="Maximum summary length in characters"
    )


class DocumentSummaryResponse(BaseModel):
    """Response model for document summary"""

    document_id: int = Field(..., description="Document ID")
    filename: str = Field(..., description="Document filename")
    summary: str = Field(..., description="Generated summary")
    model: str = Field(..., description="LLM model used")
    tokens_used: TokenUsage = Field(..., description="Token usage statistics")


class HealthCheckResponse(BaseModel):
    """Response model for health check"""

    status: str = Field(..., description="Service status")
    llm_available: bool = Field(..., description="LLM service availability")
    vector_db_available: bool = Field(..., description="Vector database availability")
    postgres_available: bool = Field(..., description="PostgreSQL availability")
