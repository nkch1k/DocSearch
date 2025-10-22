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
echo "Waiting for databases to be ready..."
sleep 5

# Create uploads directory
mkdir -p uploads

# Run the application
echo "Starting FastAPI application..."
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
