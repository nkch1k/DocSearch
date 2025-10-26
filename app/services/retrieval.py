from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.qdrant import qdrant_manager
from app.db.postgres import DocumentRepository
from app.services.embedding import embedding_service
from app.core.settings import settings
import logging

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving relevant document chunks using hybrid approach"""

    async def retrieve_context(
        self,
        question: str,
        db: AsyncSession,
        top_k: int = 5,
        score_threshold: float = 0.3,
        document_id: Optional[int] = None,
        use_postgres_fallback: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant context for a question using vector search
        with optional PostgreSQL fallback

        Args:
            question: User's question
            db: Database session
            top_k: Number of top results to retrieve
            score_threshold: Minimum similarity score
            document_id: Optional filter by specific document
            use_postgres_fallback: Use PostgreSQL as fallback if vector search fails

        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Primary: Vector search in Qdrant
            results = await self._vector_search(
                question=question,
                db=db,
                top_k=top_k,
                score_threshold=score_threshold,
                document_id=document_id
            )

            # If no results and fallback is enabled, try PostgreSQL
            if not results and use_postgres_fallback:
                logger.info("No results from vector search, trying PostgreSQL fallback")
                results = await self._postgres_fallback(
                    question=question,
                    db=db,
                    top_k=top_k,
                    document_id=document_id
                )

            return results

        except Exception as e:
            logger.error(f"Error in retrieval: {e}")

            # If vector search fails and fallback is enabled
            if use_postgres_fallback:
                logger.info("Vector search failed, using PostgreSQL fallback")
                try:
                    return await self._postgres_fallback(
                        question=question,
                        db=db,
                        top_k=top_k,
                        document_id=document_id
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    return []

            return []

    async def _vector_search(
        self,
        question: str,
        db: AsyncSession,
        top_k: int,
        score_threshold: float,
        document_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Perform vector similarity search in Qdrant

        Args:
            question: User's question
            db: Database session
            top_k: Number of results
            score_threshold: Minimum score
            document_id: Optional document filter

        Returns:
            List of relevant chunks
        """
        # Generate query embedding
        query_embedding = await embedding_service.embed_query(question)

        # Search in Qdrant
        results = await qdrant_manager.search_similar(
            query_embedding=query_embedding,
            limit=top_k,
            document_id=document_id,
            score_threshold=score_threshold
        )

        # Enrich with document metadata
        enriched_results = await self._enrich_results(results, db)

        logger.info(f"Vector search found {len(enriched_results)} results")
        return enriched_results

    async def _postgres_fallback(
        self,
        question: str,
        db: AsyncSession,
        top_k: int,
        document_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Fallback to PostgreSQL full-text search when vector search is unavailable

        Args:
            question: User's question
            db: Database session
            top_k: Number of results
            document_id: Optional document filter

        Returns:
            List of relevant chunks
        """
        repo = DocumentRepository(db)

        # If specific document requested, get its chunks
        if document_id:
            document = await repo.get_document(document_id)
            if not document:
                return []

            chunks = await qdrant_manager.get_document_chunks(document_id)

            # Simple text matching for chunks
            matched_chunks = []
            question_lower = question.lower()

            for chunk in chunks[:top_k]:
                chunk_text = chunk.get("text", "").lower()
                # Simple relevance: count matching words
                matching_words = sum(1 for word in question_lower.split()
                                   if word in chunk_text and len(word) > 2)

                if matching_words > 0:
                    matched_chunks.append({
                        "text": chunk.get("text"),
                        "score": matching_words / len(question_lower.split()),  # Simple score
                        "chunk_index": chunk.get("chunk_index"),
                        "document": {
                            "id": document_id,
                            "filename": document.filename,
                            "file_type": document.file_type
                        }
                    })

            # Sort by score
            matched_chunks.sort(key=lambda x: x["score"], reverse=True)
            return matched_chunks[:top_k]

        else:
            # Search documents by filename/content
            documents = await repo.search_documents(question)

            results = []
            for doc in documents[:top_k]:
                # Get preview or first chunk
                chunks = await qdrant_manager.get_document_chunks(doc.id)

                if chunks:
                    first_chunk = chunks[0]
                    results.append({
                        "text": first_chunk.get("text", doc.content_preview or ""),
                        "score": 0.5,  # Default score for keyword match
                        "chunk_index": 0,
                        "document": {
                            "id": doc.id,
                            "filename": doc.filename,
                            "file_type": doc.file_type
                        }
                    })

            logger.info(f"PostgreSQL fallback found {len(results)} results")
            return results

    async def _enrich_results(
        self,
        results: List[Dict],
        db: AsyncSession
    ) -> List[Dict]:
        """
        Enrich search results with document metadata from PostgreSQL

        Args:
            results: Raw results from Qdrant
            db: Database session

        Returns:
            Enriched results
        """
        if not results:
            return []

        enriched = []
        repo = DocumentRepository(db)

        for result in results:
            doc_id = result.get("document_id")
            document = await repo.get_document(doc_id)

            enriched.append({
                "score": result.get("score", 0.0),
                "text": result.get("text", ""),
                "chunk_index": result.get("chunk_index", 0),
                "document": {
                    "id": doc_id,
                    "filename": document.filename if document else "Unknown",
                    "file_type": document.file_type if document else "Unknown",
                    "upload_date": document.upload_date.isoformat() if document else None
                }
            })

        return enriched

    async def get_full_document_context(
        self,
        document_id: int,
        db: AsyncSession
    ) -> Optional[Dict]:
        """
        Get full context of a specific document

        Args:
            document_id: Document ID
            db: Database session

        Returns:
            Full document context
        """
        repo = DocumentRepository(db)
        document = await repo.get_document(document_id)

        if not document:
            return None

        # Get all chunks
        chunks = await qdrant_manager.get_document_chunks(document_id)

        # Combine all chunks
        full_text = "\n\n".join(chunk.get("text", "") for chunk in chunks)

        return {
            "document_id": document_id,
            "filename": document.filename,
            "file_type": document.file_type,
            "full_text": full_text,
            "chunks": chunks,
            "metadata": {
                "upload_date": document.upload_date.isoformat(),
                "file_size": document.file_size,
                "num_chunks": len(chunks)
            }
        }


# Singleton instance
retrieval_service = RetrievalService()
