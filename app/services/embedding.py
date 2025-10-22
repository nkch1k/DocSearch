from typing import List, Union
from sentence_transformers import SentenceTransformer
from app.core.settings import settings
import numpy as np


class EmbeddingService:
    """Service for generating embeddings"""

    def __init__(self):
        self.model = None
        self.model_name = settings.EMBEDDING_MODEL
        self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

    def encode_text(self, text: str) -> List[float]:
        """
        Encode a single text into embedding

        Args:
            text: Text to encode

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * settings.EMBEDDING_DIMENSION

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            # Convert to list and ensure it's the right type
            return embedding.tolist()
        except Exception as e:
            print(f"Error encoding text: {e}")
            raise

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts into embeddings

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            # Filter out empty texts but keep track of indices
            valid_texts = []
            valid_indices = []
            for idx, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(idx)

            if not valid_texts:
                # All texts are empty, return zero vectors
                return [[0.0] * settings.EMBEDDING_DIMENSION] * len(texts)

            # Encode valid texts
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=len(valid_texts) > 10,
                batch_size=32
            )

            # Create result list with zero vectors for empty texts
            result = []
            valid_idx = 0
            zero_vector = [0.0] * settings.EMBEDDING_DIMENSION

            for idx in range(len(texts)):
                if idx in valid_indices:
                    result.append(embeddings[valid_idx].tolist())
                    valid_idx += 1
                else:
                    result.append(zero_vector)

            return result
        except Exception as e:
            print(f"Error encoding batch: {e}")
            raise

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score between -1 and 1
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    async def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for document chunks

        Args:
            chunks: List of text chunks

        Returns:
            List of embeddings
        """
        return self.encode_batch(chunks)

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query

        Args:
            query: Search query text

        Returns:
            Query embedding
        """
        return self.encode_text(query)


# Singleton instance
embedding_service = EmbeddingService()
