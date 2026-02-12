#!/bin/bash
set -e

echo "🚀 NexusOmegaCore Bootstrap Script"
echo "=================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please copy .env.example to .env and configure it:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "✅ .env file found"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "Please start Docker and try again."
    exit 1
fi

echo "✅ Docker is running"

# Build and start services
echo ""
echo "📦 Building Docker images..."
cd infra
docker compose build

echo ""
echo "🚀 Starting services..."
docker compose up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check health
echo ""
echo "🏥 Checking health endpoint..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo "✅ Backend is healthy!"
        break
    fi
    attempt=$((attempt + 1))
    echo "Attempt $attempt/$max_attempts - waiting for backend..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Backend health check failed after $max_attempts attempts"
    echo "Check logs with: docker compose logs backend"
    exit 1
fi

echo ""
echo "✅ NexusOmegaCore is ready!"
echo ""
echo "📊 Service Status:"
docker compose ps

echo ""
echo "🔗 Endpoints:"
echo "  - Backend API: http://localhost:8000"
echo "  - Health Check: http://localhost:8000/api/v1/health"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Useful commands:"
echo "  - View logs: docker compose logs -f"
echo "  - Stop services: docker compose down"
echo "  - Restart: docker compose restart"
echo ""
