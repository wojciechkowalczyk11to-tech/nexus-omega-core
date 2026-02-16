# 🚀 NexusOmegaCore

**Telegram AI Aggregator Bot** with multi-provider LLM support, RBAC, RAG, monetization, and GitHub Devin-mode.

## 📋 Features

- **Multi-Provider AI**: Gemini, DeepSeek, Groq, OpenRouter, Grok, OpenAI, Claude
- **RBAC**: Role-based access control (DEMO, FULL_ACCESS, ADMIN)
- **Smart Routing**: Automatic difficulty classification and profile selection (ECO/SMART/DEEP)
- **Fallback Chain**: Automatic provider failover for reliability
- **RAG**: Document upload and semantic search
- **Vertex AI Search**: Integrated knowledge base with citations
- **Memory Management**: Session snapshots and absolute user memory
- **Monetization**: Telegram Stars payments with subscription tiers
- **GitHub Devin-mode**: Automated code generation and PR creation
- **Usage Tracking**: Detailed cost tracking and daily limits
- **Structured Logging**: JSON logs with request tracing

## 🏗️ Architecture

```
nexus-omega-core/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── api/v1/      # API routes
│   │   ├── core/        # Config, security, logging
│   │   ├── db/          # SQLAlchemy models
│   │   ├── services/    # Business logic
│   │   ├── providers/   # AI provider implementations
│   │   ├── tools/       # RAG, Vertex, GitHub tools
│   │   └── workers/     # Celery tasks
│   ├── alembic/         # Database migrations
│   └── tests/           # Backend tests
├── telegram_bot/         # Telegram bot client
│   ├── handlers/        # Command and message handlers
│   ├── middleware/      # Access control, rate limiting
│   └── tests/           # Bot tests
├── infra/               # Docker infrastructure
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   ├── Dockerfile.bot
│   └── Dockerfile.worker
└── scripts/             # Utility scripts
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Telegram Bot Token (from @BotFather)
- API keys for AI providers (Gemini, DeepSeek, etc.)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/wojciechkowalczyk11to-tech/nexus-omega-core.git
   cd nexus-omega-core
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Bootstrap the project**
   ```bash
   ./scripts/bootstrap.sh
   ```

4. **Verify deployment**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

### Configuration

Edit `.env` file with your credentials:

```env
# Required
TELEGRAM_BOT_TOKEN=your_bot_token
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key

# Optional providers
GROQ_API_KEY=
OPENROUTER_API_KEY=
XAI_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Access control
DEMO_UNLOCK_CODE=your_demo_code
BOOTSTRAP_ADMIN_CODE=your_admin_code
JWT_SECRET_KEY=your_256bit_secret

# Database
POSTGRES_PASSWORD=changeme
```

## 📊 Database Schema

> **Note:** The Postgres image must include pgvector (`pgvector/pgvector:pg16`) because migrations run `CREATE EXTENSION vector` for RAG embedding support.

**11 Tables:**
- `users` - User accounts with RBAC
- `chat_sessions` - Conversation sessions with snapshots
- `messages` - Message history
- `usage_ledger` - AI usage and cost tracking
- `tool_counters` - Daily tool usage limits
- `audit_logs` - Admin action tracking
- `invite_codes` - Invitation system
- `rag_items` - Uploaded documents
- `user_memories` - Persistent key-value memory
- `payments` - Telegram Stars transactions

## 🤖 Bot Commands

### User Commands
- `/start` - Welcome message
- `/help` - Command list
- `/mode <eco|smart|deep>` - Set AI profile
- `/session` - Session management
- `/memory` - Absolute memory management
- `/export` - Export conversation
- `/usage` - Usage statistics

### Admin Commands
- `/admin` - Admin panel
- `/stats` - System statistics
- `/invite` - Generate invite codes

## 🔐 RBAC Matrix

| Feature | DEMO | FULL_ACCESS | ADMIN |
|---------|------|-------------|-------|
| Gemini ECO | ✅ | ✅ | ✅ |
| DeepSeek | ✅ (50/day) | ✅ | ✅ |
| Groq | ✅ | ✅ | ✅ |
| OpenRouter | ✅ | ✅ | ✅ |
| Grok | ✅ (5/day) | ✅ | ✅ |
| Web Search | ✅ (5/day) | ✅ | ✅ |
| Smart Credits | ✅ (20/day) | ✅ | ✅ |
| OpenAI GPT-4 | ❌ | ✅ | ✅ |
| Claude | ❌ | ✅ | ✅ |
| DEEP mode | ❌ | ✅ | ✅ |
| RAG Upload | ❌ | ✅ | ✅ |
| GitHub Devin | ❌ | ✅ | ✅ |
| Daily Budget | $0 | $5 | Unlimited |

## 💳 Subscription Plans

| Plan | Stars | Duration | Features |
|------|-------|----------|----------|
| Starter | 100 | 30 days | FULL_ACCESS role |
| Pro | 250 | 30 days | Higher limits |
| Ultra | 500 | 30 days | Priority support |
| Enterprise | 1000 | 30 days | Custom limits |

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest tests/ -v --cov=app

# Bot tests
cd telegram_bot
pytest tests/ -v

# Linting
ruff check backend/ telegram_bot/
ruff format backend/ telegram_bot/
```

## 📈 Monitoring

- **Health Check**: `GET /api/v1/health`
- **API Docs**: http://localhost:8000/docs
- **Logs**: `docker compose logs -f`

## 🛠️ Development

### Running locally

```bash
# Start services
cd infra
docker compose up -d

# View logs
docker compose logs -f backend

# Restart service
docker compose restart backend

# Stop all
docker compose down
```

### Database migrations

```bash
cd backend

# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## 📝 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/unlock` - Unlock DEMO access
- `POST /api/v1/auth/bootstrap` - Bootstrap admin
- `GET /api/v1/auth/me` - Get current user

### Chat
- `POST /api/v1/chat` - Send message
- `GET /api/v1/chat/providers` - List providers

### Sessions
- `GET /api/v1/sessions` - List sessions
- `POST /api/v1/sessions` - Create session
- `DELETE /api/v1/sessions/{id}` - Delete session

### Memory
- `GET /api/v1/memory` - List memories
- `POST /api/v1/memory` - Set memory
- `DELETE /api/v1/memory/{key}` - Delete memory

### Usage
- `GET /api/v1/usage/summary` - Usage summary
- `GET /api/v1/usage/costs-by-provider` - Cost breakdown

### Admin
- `GET /api/v1/admin/stats` - System stats
- `GET /api/v1/admin/users` - List users
- `POST /api/v1/admin/invite` - Create invite code

## 🔒 Security

- JWT authentication (HS256, 24h expiration)
- SHA-256 hashed invite codes
- Rate limiting (30 req/min per user)
- RBAC enforcement at API and bot level
- Budget caps and daily limits
- Audit logging for admin actions

## 📚 Documentation

- [API Contract](docs/API_CONTRACT.md) - Full API specification
- [Runbook](docs/RUNBOOK.md) - Operations guide
- [Smoke Tests](docs/SMOKE_TESTS.md) - Testing scenarios

## 🤝 Contributing

This is a private project. Contact the maintainer for access.

## 📄 License

Proprietary - All rights reserved

## 🆘 Support

For issues and feature requests, visit: https://help.manus.im

---

**Built with ❤️ by Manus AI**
