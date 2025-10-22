from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.settings import settings
from app.db.postgres import init_db
from app.db.qdrant import qdrant_manager
from app.api import document


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    print("Starting DocSearch application...")

    # Initialize PostgreSQL database
    print("Initializing PostgreSQL database...")
    await init_db()
    print("PostgreSQL database initialized")

    # Initialize Qdrant collection
    print("Initializing Qdrant collection...")
    await qdrant_manager.init_collection()
    print("Qdrant collection initialized")

    print("Application started successfully")

    yield

    # Shutdown
    print("Shutting down DocSearch application...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Document search and retrieval system with semantic search capabilities",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "upload": "/api/documents/upload",
            "list": "/api/documents/",
            "search": "/api/documents/search/query",
            "stats": "/api/documents/stats/overview"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
