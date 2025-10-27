from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "DocSearch"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str  # Load from .env - no default for security!
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "docsearch"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "documents"
    QDRANT_API_KEY: Optional[str] = None

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # OpenAI (optional, for better embeddings)
    OPENAI_API_KEY: Optional[str] = None

    # LLM Settings
    LLM_MODEL: str = "gpt-4o-mini"  # or "gpt-4", "gpt-3.5-turbo"
    LLM_MAX_TOKENS: int = 1000
    LLM_TEMPERATURE: float = 0.7

    # File upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".md", ".txt"]
    UPLOAD_DIR: str = "./uploads"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
