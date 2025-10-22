from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from app.core.settings import settings
import uuid


class QdrantManager:
    """Manager for Qdrant vector database operations"""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    async def init_collection(self):
        """Initialize Qdrant collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                print(f"Collection '{self.collection_name}' created successfully")
            else:
                print(f"Collection '{self.collection_name}' already exists")
        except Exception as e:
            print(f"Error initializing collection: {e}")
            raise

    async def add_documents(
        self,
        document_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add document chunks with embeddings to Qdrant

        Args:
            document_id: ID of the document in PostgreSQL
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: Additional metadata for the document

        Returns:
            Number of chunks added
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "document_id": document_id,
                "chunk_index": idx,
                "text": chunk,
                "chunk_length": len(chunk)
            }

            # Add custom metadata if provided
            if metadata:
                payload.update(metadata)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )

        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        return len(points)

    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        document_id: Optional[int] = None,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar chunks

        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            document_id: Filter by specific document ID
            score_threshold: Minimum similarity score

        Returns:
            List of similar chunks with metadata
        """
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold
        )

        results = []
        for scored_point in search_result:
            results.append({
                "id": scored_point.id,
                "score": scored_point.score,
                "text": scored_point.payload.get("text"),
                "document_id": scored_point.payload.get("document_id"),
                "chunk_index": scored_point.payload.get("chunk_index"),
                "metadata": {
                    k: v for k, v in scored_point.payload.items()
                    if k not in ["text", "document_id", "chunk_index"]
                }
            })

        return results

    async def delete_document(self, document_id: int) -> bool:
        """
        Delete all chunks of a document

        Args:
            document_id: ID of the document to delete

        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            return True
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False

    async def get_document_chunks(self, document_id: int) -> List[Dict]:
        """
        Get all chunks for a specific document

        Args:
            document_id: ID of the document

        Returns:
            List of chunks with metadata
        """
        # Scroll through all points with the document_id
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1000
        )

        chunks = []
        for point in scroll_result[0]:
            chunks.append({
                "id": point.id,
                "text": point.payload.get("text"),
                "chunk_index": point.payload.get("chunk_index"),
                "chunk_length": point.payload.get("chunk_length")
            })

        # Sort by chunk index
        chunks.sort(key=lambda x: x.get("chunk_index", 0))
        return chunks

    async def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else None,
                "vectors_count": info.vectors_count if hasattr(info, 'vectors_count') else 0,
                "points_count": info.points_count if hasattr(info, 'points_count') else 0,
                "status": info.status.value if hasattr(info, 'status') else "unknown"
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}


# Singleton instance
qdrant_manager = QdrantManager()
