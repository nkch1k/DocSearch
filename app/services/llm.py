from typing import List, Dict, Optional
from openai import AsyncOpenAI
from app.core.settings import settings
import logging

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with LLM (OpenAI GPT or compatible)"""

    def __init__(self):
        self.client = None
        self.model = settings.LLM_MODEL
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client"""
        if settings.OPENAI_API_KEY:
            try:
                self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info(f"LLM client initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Error initializing LLM client: {e}")
                raise
        else:
            logger.warning("OPENAI_API_KEY not set. LLM service will not be available.")

    async def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> Dict[str, any]:
        """
        Generate an answer based on retrieved context chunks

        Args:
            question: User's question
            context_chunks: List of relevant document chunks with metadata
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 - 2.0)

        Returns:
            Dict with answer text and metadata
        """
        if not self.client:
            raise ValueError("LLM client not initialized. Please set OPENAI_API_KEY.")

        # Build context from chunks
        context_text = self._build_context(context_chunks)

        # Build system and user prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question, context_text)

        try:
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                temperature=temperature,
                top_p=0.9,
            )

            # Extract answer
            answer_text = response.choices[0].message.content

            # Extract sources
            sources = self._extract_sources(context_chunks)

            return {
                "answer": answer_text,
                "sources": sources,
                "model": self.model,
                "tokens_used": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from chunks

        Args:
            chunks: List of chunks with text and metadata

        Returns:
            Formatted context string
        """
        if not chunks:
            return "Контекст не найден."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            doc_info = chunk.get("document", {})
            filename = doc_info.get("filename", "Неизвестный документ")

            context_parts.append(
                f"[Источник {i}: {filename}]\n{text}\n"
            )

        return "\n".join(context_parts)

    def _build_system_prompt(self) -> str:
        """Build system prompt for the LLM"""
        return """Вы - профессиональный ассистент для ответов на вопросы по документам.

Ваша задача:
1. Внимательно проанализировать предоставленный контекст из документов
2. Дать максимально точный и развернутый ответ на вопрос пользователя
3. Основывать ответ ТОЛЬКО на информации из предоставленных документов
4. Если в документах нет информации для ответа, честно сказать об этом
5. Цитировать релевантные части документов при необходимости
6. Структурировать ответ логично и понятно

Правила:
- Не выдумывай информацию, которой нет в документах
- Если уверенности нет - укажи это
- Отвечай на том же языке, что и вопрос
- Будь конкретным и информативным"""

    def _build_user_prompt(self, question: str, context: str) -> str:
        """Build user prompt with question and context"""
        return f"""Контекст из документов:

{context}

---

Вопрос пользователя: {question}

Пожалуйста, дай развернутый ответ на основе предоставленного контекста."""

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract source information from chunks

        Args:
            chunks: List of chunks with metadata

        Returns:
            List of source information
        """
        sources = []
        seen_docs = set()

        for chunk in chunks:
            doc_info = chunk.get("document", {})
            doc_id = doc_info.get("id")

            # Avoid duplicate sources
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                sources.append({
                    "document_id": doc_id,
                    "filename": doc_info.get("filename", "Unknown"),
                    "file_type": doc_info.get("file_type", "Unknown"),
                    "relevance_score": chunk.get("score", 0.0)
                })

        return sources

    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a summary of the given text

        Args:
            text: Text to summarize
            max_length: Maximum length of summary

        Returns:
            Summary text
        """
        if not self.client:
            raise ValueError("LLM client not initialized.")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Создай краткое резюме следующего текста (максимум {max_length} символов)."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=max_length // 2,  # Approximate token count
                temperature=0.5
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise


# Singleton instance
llm_service = LLMService()
