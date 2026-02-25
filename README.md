# NexusOmegaCore — AI Aggregator Platform

> Intelligent multi-provider AI assistant platform built on Google Cloud Platform,
> combining the world's most advanced language models in a single, cost-optimized interface.

## Vision & Problem Statement

The AI landscape is fragmented. Users must navigate multiple subscriptions, interfaces, and pricing models to access different AI capabilities. Organizations face mounting costs managing separate API integrations with OpenAI, Google, Anthropic, xAI, and DeepSeek.

**NexusOmegaCore solves this** by providing a unified, intelligent routing layer that:
- Automatically selects the optimal AI model for each task based on complexity and cost
- Provides seamless failover between providers for 99.9% uptime
- Reduces AI costs by 40-60% through SLM-first routing strategy
- Offers a single Telegram-based interface accessible to 900M+ Telegram users

## Product Overview

### Core Features
- **Multi-LLM Aggregation**: 7 AI providers (Google Gemini, OpenAI, Anthropic Claude, xAI Grok, DeepSeek, Groq, OpenRouter) unified under one interface
- **SLM-First Cost Router**: Starts with cheapest models, escalates only when task complexity demands it
- **ReAct Agent Orchestration**: Autonomous reasoning with tool use (web search, RAG, code execution)
- **Retrieval-Augmented Generation**: Document upload + semantic search with pgvector
- **Vertex AI Search Integration**: Enterprise-grade knowledge base with citations
- **Role-Based Access Control**: DEMO → FULL_ACCESS → ADMIN with Telegram Stars monetization
- **Real-time Streaming**: Server-Sent Events for instant response delivery
- **Session Memory**: Contextual memory with session snapshots and cross-session recall

## Google Cloud Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Google Cloud Platform                         │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  Cloud Run    │    │  Cloud Run    │    │   Cloud Build     │  │
│  │  (Backend)    │◄──►│  (Bot)        │    │   (CI/CD)         │  │
│  │  FastAPI      │    │  aiogram 3    │    │                   │  │
│  └──────┬───────┘    └──────────────┘    └───────────────────┘  │
│         │                                                        │
│  ┌──────▼───────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  Cloud SQL    │    │ Memorystore  │    │  Secret Manager   │  │
│  │  PostgreSQL16 │    │ Redis 7      │    │  (API Keys)       │  │
│  │  + pgvector   │    │              │    │                   │  │
│  └──────────────┘    └──────────────┘    └───────────────────┘  │
│                                                                   │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │ Vertex AI     │    │         VPC + Cloud NAT              │   │
│  │ Search        │    │         (Private networking)         │   │
│  └──────────────┘    └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External AI Providers                          │
│  Google Gemini │ OpenAI │ Anthropic │ xAI │ DeepSeek │ Groq    │
└─────────────────────────────────────────────────────────────────┘
```

### Why Google Cloud Platform
- **Vertex AI Search**: Native integration for enterprise RAG with citations
- **Cloud Run**: Serverless auto-scaling, pay-per-use, zero infrastructure management
- **Cloud SQL + pgvector**: Managed PostgreSQL with vector similarity search
- **Memorystore**: Managed Redis for session caching and rate limiting
- **Secret Manager**: Secure API key management for 7+ AI providers
- **Cloud Build**: Integrated CI/CD with automatic deployment
- **Cloud NAT**: Secure outbound connectivity for AI provider APIs

## AI Providers & Models

| Provider | ECO (Low Cost) | SMART (Balanced) | DEEP (Premium) | Pricing (per 1M tokens) |
|----------|---------------|------------------|----------------|------------------------|
| Google Gemini | gemini-2.5-flash-preview-05-20 | gemini-2.5-flash-preview-05-20 | gemini-2.5-pro-preview-05-06 | $0.15-$10.00 |
| xAI Grok | grok-3-mini-fast | grok-3-fast | grok-3 | $0.30-$15.00 |
| OpenAI | gpt-4o-mini | gpt-4o | o3-mini | $0.15-$10.00 |
| Anthropic | claude-3-5-haiku | claude-sonnet-4 | claude-opus-4 | $0.80-$75.00 |
| DeepSeek | deepseek-chat | deepseek-chat | deepseek-reasoner | $0.14-$2.19 |
| Groq | llama-3.1-8b-instant | — | — | $0.05-$0.08 |

### SLM-First Routing Strategy
```text
User Query → Difficulty Classifier → Cost Preference Check
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
   Simple       Moderate       Complex
      │             │             │
  Tier 0:       Tier 1:       Tier 2-3:
  Ultra-Cheap   Cheap         Balanced/Premium
  (Groq/Flash)  (DeepSeek)    (GPT-4o/Claude)
      │             │             │
      └─────────────┴─────────────┘
                    │
              Response Quality Check
                    │
              ┌─────┴─────┐
              │ Adequate?  │
              └─────┬─────┘
              Yes   │   No → Escalate to next tier
              ▼     
           Return
```

## Key Differentiators

1. **Cost-Aware SLM Router** — Proprietary routing algorithm that starts with the cheapest capable model (Groq at $0.05/1M tokens) and escalates only when necessary. Average cost savings: 40-60% vs. fixed premium model usage.

2. **ReAct Agent Orchestration** — Autonomous multi-step reasoning with tool calling. The agent can search the web, query knowledge bases, and chain multiple operations — all within a Telegram conversation.

3. **Multi-Tier RBAC with Telegram Stars** — Zero-friction monetization through Telegram's native payment system. Users upgrade from DEMO (free, limited) to FULL_ACCESS (150 Stars/month) directly within Telegram.

4. **Google Cloud Native** — Built from the ground up for GCP: Vertex AI Search for enterprise RAG, Cloud Run for serverless scaling, Cloud SQL with pgvector for vector search, Secret Manager for API key security.

5. **7-Provider Fallback Chain** — If one AI provider is down, the system automatically routes to the next available provider. Zero single-point-of-failure in AI response generation.

## Technical Architecture

### Stack
- **Backend**: FastAPI 0.115 + Python 3.12 + SQLAlchemy 2.0 (async)
- **Bot**: python-telegram-bot 21.x with aiogram 3.x patterns
- **Database**: PostgreSQL 16 + pgvector extension
- **Cache**: Redis 7 (Memorystore on GCP)
- **Task Queue**: Celery 5.4 for background processing
- **AI SDKs**: google-generativeai, openai, anthropic, and OpenAI-compatible clients
- **Infrastructure**: Docker Compose (dev) / Cloud Run + Terraform (prod)
- **Monitoring**: Structured JSON logging, Cloud Logging integration
- **Security**: JWT authentication, Secret Manager, input sanitization, rate limiting

### Request Flow
```text
Telegram User
      │
      ▼
[Telegram Bot Service] ──── Rate Limiting + Auth
      │
      ▼
[Backend API (FastAPI)]
      │
      ├── Policy Engine ──── RBAC + Limits + Budget Check
      │
      ├── SLM Router ──── Difficulty → Tier → Model Selection
      │
      ├── Provider Factory ──── Create/Cache Provider Instances
      │
      ├── ReAct Orchestrator ──── Reason → Act → Observe → Think
      │        │
      │        ├── Web Search Tool
      │        ├── RAG Tool (pgvector)
      │        ├── Vertex AI Search
      │        └── GitHub Tool
      │
      └── Response → Token Counting → Cost Tracking → User
```

### Database Schema

The platform uses 13 SQLAlchemy models to manage users, conversations, AI interactions, and billing:

```text
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    users      │     │  conversations   │     │    messages       │
│──────────────│     │──────────────────│     │──────────────────│
│ id (PK)      │◄───┤ user_id (FK)     │◄───┤ conversation_id   │
│ telegram_id  │     │ title            │     │ role              │
│ role         │     │ provider         │     │ content           │
│ created_at   │     │ profile          │     │ tokens_used       │
│ is_active    │     │ created_at       │     │ cost_usd          │
└──────────────┘     └──────────────────┘     └──────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  subscriptions   │     │  usage_tracking  │     │  documents       │
│──────────────────│     │──────────────────│     │──────────────────│
│ user_id (FK)     │     │ user_id (FK)     │     │ user_id (FK)     │
│ tier             │     │ date             │     │ filename         │
│ starts_at        │     │ tokens_input     │     │ content          │
│ expires_at       │     │ tokens_output    │     │ embedding        │
│ stars_paid       │     │ cost_usd         │     │ chunk_index      │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### Security Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    Security Layers                       │
│                                                         │
│  Layer 1: Telegram Authentication                       │
│  ├── Bot token validation                               │
│  ├── User identity via Telegram ID                      │
│  └── Webhook signature verification                     │
│                                                         │
│  Layer 2: Application Security                          │
│  ├── JWT token-based API authentication                 │
│  ├── Role-Based Access Control (DEMO/FULL/ADMIN)        │
│  ├── Per-user rate limiting (Redis-backed)              │
│  └── Input sanitization and validation                  │
│                                                         │
│  Layer 3: Infrastructure Security                       │
│  ├── VPC with private networking                        │
│  ├── Cloud NAT for outbound-only connectivity           │
│  ├── Secret Manager for all API keys                    │
│  ├── Cloud SQL with private IP only                     │
│  └── IAM service accounts with least privilege          │
│                                                         │
│  Layer 4: Data Protection                               │
│  ├── Encryption at rest (Cloud SQL, Memorystore)        │
│  ├── Encryption in transit (TLS 1.3 everywhere)         │
│  ├── No PII logging in production                       │
│  └── Automatic session data expiration                  │
└─────────────────────────────────────────────────────────┘
```

### ReAct Agent Architecture

The ReAct (Reasoning + Acting) orchestrator enables autonomous multi-step problem solving:

```text
┌──────────────────────────────────────────────────────────┐
│                  ReAct Orchestrator Loop                   │
│                                                           │
│  1. THOUGHT  ──── Analyze the user's question             │
│       │                                                   │
│  2. ACTION   ──── Select and execute a tool               │
│       │           ├── web_search(query)                    │
│       │           ├── rag_search(query, top_k)            │
│       │           ├── vertex_search(query)                 │
│       │           ├── github_search(repo, query)           │
│       │           └── code_execute(snippet)                │
│       │                                                   │
│  3. OBSERVE  ──── Process tool output                     │
│       │                                                   │
│  4. THINK    ──── Is the answer complete?                 │
│       │           ├── Yes → Return final answer           │
│       │           └── No  → Loop back to step 2           │
│       │                                                   │
│  Max iterations: 5 (configurable)                         │
│  Timeout: 60 seconds per orchestration cycle              │
└──────────────────────────────────────────────────────────┘
```

### Provider Failover Strategy

When an AI provider fails, the system automatically attempts the next provider:

```text
Primary Provider (user-selected)
      │
      ▼ (on failure: timeout, 429, 500, 503)
Fallback 1: Same tier, different provider
      │
      ▼ (on failure)
Fallback 2: Lower tier, cheapest available
      │
      ▼ (on failure)
Fallback 3: Groq (always available, ultra-fast)
      │
      ▼ (all failed)
Error response with retry suggestion
```

Failover triggers:
- HTTP 429 (rate limited) — immediate switch
- HTTP 500/503 (server error) — retry once, then switch
- Timeout (>30s) — switch to faster provider
- Content filter block — switch to alternative provider

## Business Model

### Freemium via Telegram Stars

| Tier | Price | Features |
|------|-------|----------|
| DEMO (Free) | $0 | Gemini Flash, DeepSeek, Groq. 5 Grok + 5 web + 50 DeepSeek/day |
| FULL_ACCESS Monthly | 150 Stars (~$3.75) | All 7 providers, DEEP mode, RAG, GitHub, unlimited usage up to $5/day |
| FULL_ACCESS Weekly | 50 Stars (~$1.25) | Same as monthly, 7-day access |
| DEEP Day Pass | 25 Stars (~$0.63) | DEEP profile unlock for 24h |

### Enterprise (Roadmap)
- Self-hosted deployment for organizations
- White-label Telegram bot
- Custom model routing policies
- Dedicated Cloud SQL instance

## Market Opportunity

### Total Addressable Market
- Global AI assistant market: **$32B** (2025), growing at 34% CAGR
- Telegram users: **900M+** monthly active users
- AI API spending: **$4.6B** (2025), fragmented across providers

### Serviceable Market
- Telegram-first AI users in CIS/MENA/Asia: **~50M** potential users
- Cost-conscious AI power users: **~5M** globally seeking multi-provider access

### Beachhead
- Polish and Eastern European tech communities on Telegram
- Developers needing quick AI access without context-switching

## Roadmap

| Quarter | Milestone |
|---------|-----------|
| Q1 2026 | GCP production deploy, Cloud Run + Cloud SQL, Terraform IaC |
| Q2 2026 | Mobile companion app (Flutter), Firebase Auth integration |
| Q3 2026 | Enterprise self-hosted package, white-label bot toolkit |
| Q4 2026 | AI tool marketplace, community plugins, multi-language UI |
| 2027 | Voice-first interface, video processing, multi-modal agents |

### Detailed Roadmap

#### Q1 2026 — Production Launch
- [x] Complete backend API with all 7 AI providers
- [x] Telegram bot with full command set
- [x] SLM-first routing algorithm
- [x] ReAct agent orchestration
- [x] RAG pipeline with pgvector
- [x] Telegram Stars payment integration
- [ ] Cloud Run production deployment
- [ ] Terraform IaC for all GCP resources
- [ ] Cloud Build CI/CD pipeline
- [ ] Load testing and performance optimization

#### Q2 2026 — Mobile & Auth
- [ ] Flutter mobile companion app
- [ ] Firebase Authentication integration
- [ ] Push notifications for long-running tasks
- [ ] Offline mode with cached responses
- [ ] App Store and Google Play release

#### Q3 2026 — Enterprise
- [ ] Self-hosted deployment package (Docker + Helm)
- [ ] White-label bot configuration toolkit
- [ ] Organization management dashboard
- [ ] Custom model routing policies per org
- [ ] SSO integration (SAML, OIDC)
- [ ] Audit logging and compliance reports

#### Q4 2026 — Platform
- [ ] AI tool marketplace for community plugins
- [ ] Plugin SDK with sandboxed execution
- [ ] Multi-language UI (10+ languages)
- [ ] Advanced analytics dashboard
- [ ] A/B testing framework for model routing

## Why Google Cloud

1. **Vertex AI Integration** — Native search and RAG capabilities with enterprise-grade citation support, directly integrated into our knowledge pipeline.

2. **Cloud Run Serverless** — Pay only for actual compute time. Our traffic is bursty (Telegram messages), making serverless ideal vs. always-on VMs.

3. **Startup Credits** — Google for Startups Cloud Program provides credits that directly accelerate our go-to-market timeline.

4. **Global Edge Network** — Low-latency AI responses for our global Telegram user base via Google's 187+ points of presence.

5. **Managed Services** — Cloud SQL, Memorystore, Secret Manager reduce our DevOps burden to near-zero, letting us focus on product.

## Getting Started

### Prerequisites
- Docker and Docker Compose v2+
- Python 3.12+ (for local development without Docker)
- PostgreSQL 16 with pgvector extension (or use Docker)
- Redis 7+ (or use Docker)
- At least one AI provider API key

### Local Development (3 commands)
```bash
# 1. Clone and configure
git clone https://github.com/wojciechkowalczyk11to-tech/nexus-omega-core.git
cd nexus-omega-core && cp .env.example .env
# Edit .env with your API keys

# 2. Start all services
docker compose -f docker-compose.production.yml up -d

# 3. Verify
curl http://localhost:8000/api/v1/health
```

### Production Deploy (GCP)
See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for full GCP deployment instructions using:
- `infra/gcp/terraform/main.tf` — Infrastructure as Code
- `infra/gcp/cloudbuild.yaml` — CI/CD pipeline
- `infra/gcp/cloud-run-*.yaml` — Service definitions

### Environment Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Fill in your API keys (minimum one provider):
   ```bash
   # Required
   TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
   
   # At least one AI provider
   GEMINI_API_KEY=your_google_ai_key
   
   # Optional but recommended
   OPENAI_API_KEY=your_openai_key
   XAI_API_KEY=your_xai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   DEEPSEEK_API_KEY=your_deepseek_key
   ```

3. Start the services:
   ```bash
   docker compose -f docker-compose.production.yml up -d
   ```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/chat` | Send chat message |
| GET | `/api/v1/chat/stream` | SSE streaming chat |
| GET | `/api/v1/models` | List available models |
| GET | `/api/v1/user/profile` | Get user profile |
| POST | `/api/v1/documents/upload` | Upload document for RAG |
| POST | `/api/v1/payments/create` | Create Telegram Stars payment |
| GET | `/docs` | OpenAPI documentation |
| GET | `/redoc` | ReDoc documentation |

## Configuration

All configuration via environment variables (see `.env.example`):

| Category | Variables | Description |
|----------|-----------|-------------|
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_MODE` | Bot authentication and mode |
| Auth | `JWT_SECRET_KEY`, `DEMO_UNLOCK_CODE` | Security tokens |
| Database | `DATABASE_URL`, `REDIS_URL` | PostgreSQL + Redis connections |
| AI Providers | `GEMINI_API_KEY`, `OPENAI_API_KEY`, etc. | Provider API keys (7 providers) |
| GCP | `VERTEX_PROJECT_ID`, `VERTEX_LOCATION` | Google Cloud settings |
| Limits | `DEMO_GROK_DAILY`, `FULL_DAILY_USD_CAP` | Usage rate limiting |
| Features | `PAYMENTS_ENABLED`, `VERTEX_ENABLED` | Feature flags |

## Project Structure

```
nexus-omega-core/
├── backend/                    # FastAPI backend service
│   ├── app/
│   │   ├── api/v1/            # REST API endpoints
│   │   │   ├── routes_chat.py         # Chat and streaming
│   │   │   ├── routes_models.py       # Model listing
│   │   │   ├── routes_documents.py    # RAG document upload
│   │   │   ├── routes_payments.py     # Telegram Stars billing
│   │   │   └── routes_health.py       # Health checks
│   │   ├── core/              # Config, security, logging
│   │   │   ├── config.py              # Environment configuration
│   │   │   ├── security.py            # JWT, auth middleware
│   │   │   └── logging_config.py      # Structured logging
│   │   ├── db/models/         # 13 SQLAlchemy models
│   │   │   ├── user.py                # User + roles
│   │   │   ├── conversation.py        # Conversations
│   │   │   ├── message.py             # Messages + token tracking
│   │   │   ├── subscription.py        # Payment subscriptions
│   │   │   └── document.py            # RAG documents + embeddings
│   │   ├── providers/         # 7 AI provider implementations
│   │   │   ├── base_provider.py       # Abstract base class
│   │   │   ├── gemini_provider.py     # Google Gemini
│   │   │   ├── openai_provider.py     # OpenAI GPT
│   │   │   ├── claude_provider.py     # Anthropic Claude
│   │   │   ├── grok_provider.py       # xAI Grok
│   │   │   ├── deepseek_provider.py   # DeepSeek
│   │   │   ├── groq_provider.py       # Groq (fast inference)
│   │   │   └── openrouter_provider.py # OpenRouter (fallback)
│   │   ├── services/          # Business logic
│   │   │   ├── orchestrator.py        # ReAct agent orchestrator
│   │   │   ├── policy_engine.py       # RBAC + usage limits
│   │   │   ├── slm_router.py          # SLM-first model routing
│   │   │   └── provider_factory.py    # Provider instantiation
│   │   ├── tools/             # External tools
│   │   │   ├── rag_tool.py            # pgvector RAG search
│   │   │   ├── vertex_search.py       # Vertex AI Search
│   │   │   ├── web_search.py          # Web search integration
│   │   │   └── github_tool.py         # GitHub API search
│   │   └── workers/           # Celery async tasks
│   │       ├── celery_app.py          # Celery configuration
│   │       └── tasks.py               # Background task definitions
│   ├── alembic/               # Database migrations
│   └── tests/                 # Unit + integration tests
├── telegram_bot/              # Telegram bot client
│   ├── handlers/              # 9 command/message handlers
│   │   ├── start.py                   # /start command
│   │   ├── chat.py                    # Message handling
│   │   ├── model.py                   # /model selection
│   │   ├── profile.py                 # /profile command
│   │   ├── pay.py                     # /pay subscription
│   │   ├── search.py                  # /search web search
│   │   ├── rag.py                     # /rag document Q&A
│   │   ├── admin.py                   # Admin commands
│   │   └── help.py                    # /help command
│   ├── middleware/            # Auth, rate limiting
│   └── services/             # Backend API client
├── infra/                     # Infrastructure
│   ├── gcp/                  # GCP-specific configs
│   │   ├── terraform/        # Terraform IaC
│   │   │   ├── main.tf               # Resource definitions
│   │   │   ├── variables.tf           # Input variables
│   │   │   └── outputs.tf            # Output values
│   │   ├── cloudbuild.yaml   # CI/CD pipeline
│   │   └── cloud-run-*.yaml  # Service definitions
│   ├── Dockerfile.backend    # Backend container
│   ├── Dockerfile.bot        # Bot container
│   └── Dockerfile.worker     # Celery worker container
├── docs/                      # Documentation
│   └── MODELS.md             # AI models reference
├── tests/                     # Root-level test suite
│   ├── test_config.py        # Configuration tests
│   └── test_secrets_audit.py # Security audit tests
├── scripts/                   # Deployment & maintenance scripts
├── docker-compose.production.yml  # Production Docker Compose
├── ruff.toml                  # Python linter config
└── README.md                  # This file
```

## Team

**Wojciech Kowalczyk** — Solo Founder & Full-Stack AI Engineer
- Architecture design, backend development, AI integration, infrastructure
- Background in Python, machine learning, cloud architecture
- Contact: [GitHub](https://github.com/wojciechkowalczyk11to-tech)

## Contributing

This project is currently in private development. If you're interested in contributing:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please follow:
- [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
- Python code style enforced by `ruff` (see `ruff.toml`)
- All new features must include tests
- All AI provider integrations must follow `BaseProvider` interface

## Links

- **GitHub**: [github.com/wojciechkowalczyk11to-tech/nexus-omega-core](https://github.com/wojciechkowalczyk11to-tech/nexus-omega-core)
- **Deployment Guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **API Documentation**: Available at `/docs` when backend is running

## License

This project is proprietary software. All rights reserved.
