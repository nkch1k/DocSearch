from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, DateTime, BigInteger, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.future import select
from app.core.settings import settings

Base = declarative_base()


class Document(Base):
    """Document metadata model"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(BigInteger, nullable=False)  # in bytes
    file_type = Column(String(10), nullable=False)  # pdf, md, txt
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    num_chunks = Column(Integer, default=0)
    content_preview = Column(Text, nullable=True)  # first 500 chars
    status = Column(String(20), default="processing")  # processing, completed, failed

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "file_type": self.file_type,
            "upload_date": self.upload_date.isoformat() if self.upload_date else None,
            "num_chunks": self.num_chunks,
            "content_preview": self.content_preview,
            "status": self.status
        }


# Database engine and session
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


class DocumentRepository:
    """Repository for document operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_document(
        self,
        filename: str,
        file_path: str,
        file_size: int,
        file_type: str,
        content_preview: Optional[str] = None
    ) -> Document:
        """Create a new document record"""
        document = Document(
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_type,
            content_preview=content_preview
        )
        self.session.add(document)
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def get_document(self, document_id: int) -> Optional[Document]:
        """Get document by ID"""
        result = await self.session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def get_all_documents(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[Document]:
        """Get all documents with pagination"""
        result = await self.session.execute(
            select(Document)
            .order_by(Document.upload_date.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    async def update_document_status(
        self,
        document_id: int,
        status: str,
        num_chunks: Optional[int] = None
    ) -> Optional[Document]:
        """Update document status and chunk count"""
        document = await self.get_document(document_id)
        if document:
            document.status = status
            if num_chunks is not None:
                document.num_chunks = num_chunks
            await self.session.commit()
            await self.session.refresh(document)
        return document

    async def delete_document(self, document_id: int) -> bool:
        """Delete document by ID"""
        document = await self.get_document(document_id)
        if document:
            await self.session.delete(document)
            await self.session.commit()
            return True
        return False

    async def search_documents(self, query: str) -> List[Document]:
        """Search documents by filename"""
        result = await self.session.execute(
            select(Document)
            .where(Document.filename.ilike(f"%{query}%"))
            .order_by(Document.upload_date.desc())
        )
        return result.scalars().all()
