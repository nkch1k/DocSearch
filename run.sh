#!/bin/bash

# DocSearch startup script

echo "Starting DocSearch application..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo ".env file created. Please edit it with your configuration."
fi

# Start databases
echo "Starting PostgreSQL and Qdrant with Docker..."
docker-compose up -d

# Wait for databases to be ready
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h localhost -p 5432 > /dev/null 2>&1; do
    echo "  PostgreSQL not ready yet, waiting..."
    sleep 1
done
echo "PostgreSQL is ready."

echo "Waiting for Qdrant to be ready..."
until curl -sf http://localhost:6333/health > /dev/null; do
    echo "  Qdrant not ready yet, waiting..."
    sleep 1
done
echo "Qdrant is ready."
# Create uploads directory
mkdir -p uploads

# Run the application
echo "Starting FastAPI application..."
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
