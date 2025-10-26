# DocSearch

Document search and retrieval system with semantic search capabilities.

## Features

- Upload documents (PDF, MD, TXT)
- Automatic text extraction and parsing
- Intelligent text chunking with overlap
- Vector embeddings generation
- Semantic search using Qdrant
- Metadata storage in PostgreSQL
- **RAG (Retrieval-Augmented Generation)** - AI-powered question answering
- **LLM Integration** - OpenAI GPT models for answer generation
- Hybrid retrieval (vector search + PostgreSQL fallback)
- RESTful API with FastAPI

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ HTTP/REST
       │
┌──────▼──────────────────────────────────┐
│         FastAPI Application              │
│  ┌────────────────────────────────────┐ │
│  │     API Layer (document.py)        │ │
│  └───────────┬────────────────────────┘ │
│              │                           │
│  ┌───────────▼────────────┐             │
│  │   Services Layer       │             │
│  │  - Parse (parse.py)    │             │
│  │  - Embedding           │             │
│  └───────────┬────────────┘             │
│              │                           │
│  ┌───────────▼────────────┐             │
│  │    Database Layer      │             │
│  │  - PostgreSQL          │             │
│  │  - Qdrant              │             │
│  └────────────────────────┘             │
└─────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for databases)
- PostgreSQL 15+
- Qdrant

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd DocSearch
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env file:
```bash
cp .env.example .env
# Edit .env with your settings
```

5. Start databases with Docker:
```bash
docker-compose up -d
```

6. Run the application:
```bash
python -m uvicorn app.main:app --reload
```

Or:
```bash
python app/main.py
```

## API Endpoints

### Upload Document
```bash
POST /api/documents/upload
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### List Documents
```bash
GET /api/documents/?skip=0&limit=100

curl "http://localhost:8000/api/documents/"
```

### Get Document
```bash
GET /api/documents/{document_id}

curl "http://localhost:8000/api/documents/1"
```

### Get Document Chunks
```bash
GET /api/documents/{document_id}/chunks

curl "http://localhost:8000/api/documents/1/chunks"
```

### Semantic Search
```bash
GET /api/documents/search/query?query=your+search+query&limit=5

curl "http://localhost:8000/api/documents/search/query?query=machine+learning&limit=5"
```

### Search by Filename
```bash
GET /api/documents/search/by-name?query=filename

curl "http://localhost:8000/api/documents/search/by-name?query=report"
```

### Delete Document
```bash
DELETE /api/documents/{document_id}

curl -X DELETE "http://localhost:8000/api/documents/1"
```

### Get Statistics
```bash
GET /api/documents/stats/overview

curl "http://localhost:8000/api/documents/stats/overview"
```

---

## RAG (Question Answering) Endpoints

### Ask Question
Ask a question and get an AI-generated answer based on your documents:

```bash
POST /api/answer/ask
Content-Type: application/json

curl -X POST "http://localhost:8000/api/answer/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Что такое машинное обучение?",
    "top_k": 5,
    "score_threshold": 0.3,
    "temperature": 0.7
  }'
```

Response:
```json
{
  "question": "Что такое машинное обучение?",
  "answer": "Машинное обучение - это...",
  "sources": [
    {
      "document_id": 1,
      "filename": "ml_guide.pdf",
      "file_type": "pdf",
      "relevance_score": 0.85
    }
  ],
  "chunks_used": [...],
  "model": "gpt-4o-mini",
  "tokens_used": {
    "prompt": 450,
    "completion": 200,
    "total": 650
  },
  "retrieval_method": "vector_search",
  "processing_time_ms": 1234.56
}
```

### Summarize Document
Generate a summary of a specific document:

```bash
POST /api/answer/summarize
Content-Type: application/json

curl -X POST "http://localhost:8000/api/answer/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": 1,
    "max_length": 200
  }'
```

### Health Check
Check the health of RAG system components:

```bash
GET /api/answer/health

curl "http://localhost:8000/api/answer/health"
```

Response:
```json
{
  "status": "healthy",
  "llm_available": true,
  "vector_db_available": true,
  "postgres_available": true
}
```

### Get Document Context
Get full context of a document (for debugging):

```bash
GET /api/answer/context/{document_id}

curl "http://localhost:8000/api/answer/context/1"
```

## Configuration

Configuration is managed through environment variables in `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | PostgreSQL username | postgres |
| `POSTGRES_PASSWORD` | PostgreSQL password | postgres |
| `POSTGRES_HOST` | PostgreSQL host | localhost |
| `POSTGRES_PORT` | PostgreSQL port | 5432 |
| `POSTGRES_DB` | PostgreSQL database | docsearch |
| `QDRANT_HOST` | Qdrant host | localhost |
| `QDRANT_PORT` | Qdrant port | 6333 |
| `QDRANT_COLLECTION_NAME` | Qdrant collection name | documents |
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 |
| `EMBEDDING_DIMENSION` | Embedding vector size | 384 |
| `CHUNK_SIZE` | Text chunk size | 500 |
| `CHUNK_OVERLAP` | Chunk overlap size | 50 |
| `OPENAI_API_KEY` | OpenAI API key (required for RAG) | - |
| `LLM_MODEL` | OpenAI model to use | gpt-4o-mini |
| `LLM_MAX_TOKENS` | Max tokens in LLM response | 1000 |
| `LLM_TEMPERATURE` | LLM temperature (0-2) | 0.7 |
| `MAX_FILE_SIZE` | Max upload size (bytes) | 10485760 (10MB) |

## Development

### Project Structure

```
DocSearch/
├── app/
│   ├── api/
│   │   ├── document.py      # Document API endpoints
│   │   └── answer.py        # RAG/QA API endpoints
│   ├── core/
│   │   ├── config.py
│   │   └── settings.py      # Application settings
│   ├── db/
│   │   ├── postgres.py      # PostgreSQL operations
│   │   └── qdrant.py        # Qdrant operations
│   ├── models/
│   │   ├── document.py      # Document models
│   │   └── answer.py        # Answer/RAG models
│   ├── services/
│   │   ├── embedding.py     # Embedding generation
│   │   ├── llm.py           # LLM integration (OpenAI)
│   │   ├── retrieval.py     # Context retrieval
│   │   └── parse.py         # Document parsing
│   └── main.py              # FastAPI application
├── tests/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### Running Tests

```bash
pytest tests/
```

## API Documentation

Once the application is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT
