from typing import List, Tuple
import re
from pathlib import Path
from PyPDF2 import PdfReader
from app.core.settings import settings


class DocumentParser:
    """Parser for different document types"""

    @staticmethod
    def parse_txt(file_path: str) -> str:
        """Parse TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            return content

    @staticmethod
    def parse_md(file_path: str) -> str:
        """Parse Markdown file"""
        # Markdown is just text, so we can read it directly
        return DocumentParser.parse_txt(file_path)

    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """Parse PDF file"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error parsing PDF: {str(e)}")

    @staticmethod
    def parse_document(file_path: str, file_type: str) -> str:
        """
        Parse document based on file type

        Args:
            file_path: Path to the file
            file_type: Type of file (pdf, md, txt)

        Returns:
            Extracted text content
        """
        file_type = file_type.lower().replace(".", "")

        if file_type == "txt":
            return DocumentParser.parse_txt(file_path)
        elif file_type == "md":
            return DocumentParser.parse_md(file_path)
        elif file_type == "pdf":
            return DocumentParser.parse_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Strip whitespace
        text = text.strip()
        return text

    @staticmethod
    def split_into_chunks(
        text: str,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[str]:
        """
        Split text into chunks with overlap

        Args:
            text: Text to split
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = settings.CHUNK_OVERLAP

        if len(text) == 0:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If this is not the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                search_start = max(start, end - 100)
                search_text = text[search_start:end + 100]

                # Find sentence boundaries (., !, ?, \n)
                sentence_ends = [
                    m.end() + search_start
                    for m in re.finditer(r'[.!?\n]\s+', search_text)
                ]

                if sentence_ends:
                    # Find the closest sentence end to our target end position
                    closest_end = min(sentence_ends, key=lambda x: abs(x - end))
                    if abs(closest_end - end) < 100:  # Only use if close enough
                        end = closest_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position considering overlap
            start = end - chunk_overlap

            # Prevent infinite loop
            if end - chunk_overlap <= start:
                start = end

        return chunks

    @staticmethod
    def get_preview(text: str, max_length: int = 500) -> str:
        """Get preview of text"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


class DocumentProcessor:
    """High-level document processor"""

    def __init__(self):
        self.parser = DocumentParser()

    async def process_document(
        self,
        file_path: str,
        file_type: str
    ) -> Tuple[str, List[str], str]:
        """
        Process document: parse, clean, and chunk

        Args:
            file_path: Path to the file
            file_type: Type of file

        Returns:
            Tuple of (full_text, chunks, preview)
        """
        # Parse document
        text = self.parser.parse_document(file_path, file_type)

        # Clean text
        clean_text = self.parser.clean_text(text)

        # Split into chunks
        chunks = self.parser.split_into_chunks(clean_text)

        # Get preview
        preview = self.parser.get_preview(clean_text)

        return clean_text, chunks, preview


# Singleton instance
document_processor = DocumentProcessor()
