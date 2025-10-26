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
            return ">=B5:AB =5 =0945=."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            doc_info = chunk.get("document", {})
            filename = doc_info.get("filename", "58725AB=K9 4>:C<5=B")

            context_parts.append(
                f"[AB>G=8: {i}: {filename}]\n{text}\n"
            )

        return "\n".join(context_parts)

    def _build_system_prompt(self) -> str:
        """Build system prompt for the LLM"""
        return """"K - ?@>D5AA8>=0;L=K9 0AA8AB5=B 4;O >B25B>2 =0 2>?@>AK ?> 4>:C<5=B0<.

"2>O 7040G0:
1. =8<0B5;L=> ?@>0=0;878@>20BL ?@54>AB02;5==K9 :>=B5:AB 87 4>:C<5=B>2
2. 0BL <0:A8<0;L=> B>G=K9 8 @0725@=CBK9 >B25B =0 2>?@>A ?>;L7>20B5;O
3. A=>2K20BL >B25B ", =0 8=D>@<0F88 87 ?@54>AB02;5==KE 4>:C<5=B>2
4. A;8 2 4>:C<5=B0E =5B 8=D>@<0F88 4;O >B25B0, G5AB=> A:070BL >1 MB><
5. &8B8@>20BL @5;520=B=K5 G0AB8 4>:C<5=B>2 ?@8 =5>1E>48<>AB8
6. !B@C:BC@8@>20BL >B25B ;>38G=> 8 ?>=OB=>

@028;0:
-  2K4C<K209 8=D>@<0F8N, :>B>@>9 =5B 2 4>:C<5=B0E
- A;8 C25@5==>AB8 =5B - C:068 MB>
- B25G09 =0 B>< 65 O7K:5, GB> 8 2>?@>A
- C4L :>=:@5B=K< 8 8=D>@<0B82=K<
"""

    def _build_user_prompt(self, question: str, context: str) -> str:
        """
        Build user prompt with question and context

        Args:
            question: User's question
            context: Context from documents

        Returns:
            Formatted prompt
        """
        return f""">=B5:AB 87 4>:C<5=B>2:

{context}

---

>?@>A ?>;L7>20B5;O: {question}

>60;C9AB0, 409 @0725@=CBK9 >B25B =0 >A=>25 ?@54>AB02;5==>3> :>=B5:AB0."""

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
                        "content": f"!>7409 :@0B:>5 @57N<5 A;54CNI53> B5:AB0 (<0:A8<C< {max_length} A8<2>;>2)."
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
