# NexusOmegaCore - Deployment Guide

Complete step-by-step guide to deploy NexusOmegaCore Telegram AI aggregator bot.

## 📋 Prerequisites

- Docker + Docker Compose
- Git
- Telegram Bot Token (from @BotFather)
- API Keys for AI providers (at least 1)

## 🚀 Quick Start (5 minutes)

### 1. Clone Repository

```bash
git clone https://github.com/wojciechkowalczyk11to-tech/nexus-omega-core.git
cd nexus-omega-core
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in **required** fields:

```env
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
DEMO_UNLOCK_CODE=DEMO2024
BOOTSTRAP_ADMIN_CODE=ADMIN2024
JWT_SECRET_KEY=generate_random_32_chars_here
POSTGRES_PASSWORD=change_this_in_production

# At least ONE AI provider API key
GEMINI_API_KEY=your_gemini_key_here
# OR
GROQ_API_KEY=your_groq_key_here
# OR
DEEPSEEK_API_KEY=your_deepseek_key_here
```

### 3. Bootstrap Database

```bash
./scripts/bootstrap.sh
```

This will:
- Create database schema
- Run Alembic migrations
- Set up initial data

### 4. Start Services

```bash
docker compose -f infra/docker-compose.yml up -d
```

This starts:
- PostgreSQL (port 5432)
- Redis (port 6379)
- Backend API (port 8000)
- Telegram Bot
- Celery Worker

### 5. Verify Deployment

```bash
# Check health endpoint
curl http://localhost:8000/api/v1/health

# Expected response:
{
  "status": "healthy",
  "database": "healthy",
  "redis": "healthy"
}

# Check logs
docker compose -f infra/docker-compose.yml logs -f telegram_bot
```

### 6. Test Bot

Open Telegram and:
1. Find your bot (@your_bot_username)
2. Send `/start`
3. Send `/unlock DEMO2024`
4. Send a message: "Wyjaśnij AI"

✅ **Done!** Your bot is live.

## 🧪 Docker Smoke (CI parity)

Run the same deterministic smoke flow locally:

```bash
cp .env.example .env
echo "TELEGRAM_DRY_RUN=1" >> .env
docker system prune -af || true
docker builder prune -af || true
df -h
docker compose -f infra/docker-compose.yml up --build -d
python3 scripts/wait_for_backend_health.py --url http://localhost:8000/api/v1/health --timeout 180
docker compose -f infra/docker-compose.yml ps
docker compose -f infra/docker-compose.yml exec -T backend alembic upgrade head
docker compose -f infra/docker-compose.yml down -v
```

You can also run the health waiter independently:

```bash
python3 scripts/wait_for_backend_health.py --url http://localhost:8000/api/v1/health --timeout 180
```

`TELEGRAM_DRY_RUN` is optional and defaults to disabled (`0`/unset). Set `TELEGRAM_DRY_RUN=1` only for CI/smoke runs to start the bot process without contacting Telegram network APIs.

## 🔑 API Keys Setup

### Required (at least 1)

**Google Gemini** (Free tier, recommended)
1. Go to https://makersuite.google.com/app/apikey
2. Create API key
3. Add to `.env`: `GEMINI_API_KEY=...`

**Groq** (Free tier, fast)
1. Go to https://console.groq.com/keys
2. Create API key
3. Add to `.env`: `GROQ_API_KEY=...`

### Optional (for FULL_ACCESS features)

**DeepSeek** (Paid, $0.14-$2.19/1M tokens)
1. Go to https://platform.deepseek.com/api_keys
2. Create API key
3. Add to `.env`: `DEEPSEEK_API_KEY=...`

**OpenAI** (Paid, GPT-4)
1. Go to https://platform.openai.com/api-keys
2. Create API key
3. Add to `.env`: `OPENAI_API_KEY=...`

**Anthropic Claude** (Paid, Sonnet)
1. Go to https://console.anthropic.com/settings/keys
2. Create API key
3. Add to `.env`: `ANTHROPIC_API_KEY=...`

**xAI Grok** (Paid, $5-$15/1M tokens)
1. Go to https://console.x.ai/
2. Create API key
3. Add to `.env`: `XAI_API_KEY=...`

**OpenRouter** (Free tier)
1. Go to https://openrouter.ai/keys
2. Create API key
3. Add to `.env`: `OPENROUTER_API_KEY=...`

### Tools (Optional)

**Brave Search** (Web search)
1. Go to https://brave.com/search/api/
2. Create API key
3. Add to `.env`: `BRAVE_SEARCH_API_KEY=...`

**Vertex AI Search** (Knowledge base)
1. Create GCP project
2. Enable Discovery Engine API
3. Create data store
4. Add to `.env`:
   ```
   VERTEX_PROJECT_ID=your-project-id
   VERTEX_SEARCH_DATASTORE_ID=your-datastore-id
   ```

## 🔐 Security Configuration

### Generate Secret Key

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy output to `.env`:
```env
JWT_SECRET_KEY=your_generated_secret_here
```

### Set Unlock Codes

```env
DEMO_UNLOCK_CODE=DEMO2024        # For DEMO access
BOOTSTRAP_ADMIN_CODE=ADMIN2024   # For first admin user
```

**⚠️ Change these in production!**

### Database Password

```env
POSTGRES_PASSWORD=change_this_in_production
```

## 📦 Docker Compose Services

### Services Overview

```yaml
services:
  postgres:    # PostgreSQL 16 database
  redis:       # Redis 7 cache
  backend:     # FastAPI backend API
  telegram_bot: # Telegram bot
  worker:      # Celery background worker
```

### Service Management

```bash
# Start all services
docker compose -f infra/docker-compose.yml up -d

# Stop all services
docker compose -f infra/docker-compose.yml down

# Restart specific service
docker compose -f infra/docker-compose.yml restart backend

# View logs
docker compose -f infra/docker-compose.yml logs -f telegram_bot

# Scale workers
docker compose -f infra/docker-compose.yml up -d --scale worker=3
```

## 🗄 Database Management

### Run Migrations

```bash
cd backend
alembic upgrade head
```

### Create New Migration

```bash
cd backend
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

### Backup Database

```bash
docker exec nexus-postgres pg_dump -U jarvis jarvis > backup.sql
```

### Restore Database

```bash
docker exec -i nexus-postgres psql -U jarvis jarvis < backup.sql
```

## 👤 User Management

### Create Admin User

1. Start bot: `/start`
2. Bootstrap admin: `/bootstrap ADMIN2024`
3. Verify: `/help` (should see admin commands)

### Generate Invite Codes

Use backend API or database:

```sql
INSERT INTO invite_codes (code, max_uses, expires_at, created_by_user_id)
VALUES ('INVITE123', 10, NOW() + INTERVAL '30 days', 1);
```

### Check User Status

```bash
docker exec -it nexus-postgres psql -U jarvis jarvis
```

```sql
SELECT telegram_id, username, role, authorized, credits_balance
FROM users
ORDER BY created_at DESC
LIMIT 10;
```

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Logs

```bash
# All services
docker compose -f infra/docker-compose.yml logs -f

# Specific service
docker compose -f infra/docker-compose.yml logs -f backend

# Last 100 lines
docker compose -f infra/docker-compose.yml logs --tail=100 telegram_bot
```

### Resource Usage

```bash
docker stats
```

## 🔧 Troubleshooting

### Bot Not Responding

1. Check bot logs:
   ```bash
   docker compose -f infra/docker-compose.yml logs telegram_bot
   ```

2. Verify token:
   ```bash
   echo $TELEGRAM_BOT_TOKEN
   ```

3. Test backend:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

### Database Connection Error

1. Check PostgreSQL:
   ```bash
   docker compose -f infra/docker-compose.yml ps postgres
   ```

2. Test connection:
   ```bash
   docker exec nexus-postgres pg_isready -U nexus
   ```

3. Check credentials in `.env`

### Provider API Errors

1. Verify API keys in `.env`
2. Check provider status:
   ```bash
   curl -H "Authorization: Bearer YOUR_JWT" \
        http://localhost:8000/api/v1/chat/providers
   ```

3. Test specific provider:
   ```python
   from app.providers.factory import ProviderFactory
   provider = ProviderFactory.create("gemini")
   print(provider.is_available())
   ```

### Redis Connection Error

1. Check Redis:
   ```bash
   docker compose -f infra/docker-compose.yml ps redis
   ```

2. Test connection:
   ```bash
   docker exec nexus-redis redis-cli ping
   ```

## 🚀 Production Deployment

### Environment Variables

```env
# Production settings
ENVIRONMENT=production
LOG_LEVEL=WARNING

# Database (use managed service)
DATABASE_URL=postgresql+asyncpg://user:pass@db.example.com:5432/nexus

# Redis (use managed service)
REDIS_URL=redis://cache.example.com:6379/0

# Security
JWT_SECRET_KEY=use_strong_random_key_here
DEMO_UNLOCK_CODE=change_this
BOOTSTRAP_ADMIN_CODE=change_this

# CORS (if using web frontend)
CORS_ORIGINS=["https://yourdomain.com"]
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### SSL/TLS (Let's Encrypt)

```bash
sudo certbot --nginx -d api.yourdomain.com
```

### Process Manager (systemd)

Create `/etc/systemd/system/nexus-backend.service`:

```ini
[Unit]
Description=NexusOmegaCore Backend
After=network.target

[Service]
Type=simple
User=nexus
WorkingDirectory=/opt/nexus-omega-core
ExecStart=/usr/local/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable nexus-backend
sudo systemctl start nexus-backend
```

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale backend
docker compose -f infra/docker-compose.yml up -d --scale backend=3

# Scale workers
docker compose -f infra/docker-compose.yml up -d --scale worker=5
```

### Load Balancing

Use Nginx or HAProxy:

```nginx
upstream backend {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}
```

### Database Optimization

1. Enable connection pooling (already configured)
2. Add read replicas for high traffic
3. Use pgBouncer for connection management

## 🔄 Updates

### Pull Latest Changes

```bash
git pull origin master
```

### Rebuild Containers

```bash
docker compose -f infra/docker-compose.yml build
docker compose -f infra/docker-compose.yml up -d
```

### Run New Migrations

```bash
cd backend
alembic upgrade head
```

## 📞 Support

- Repository: https://github.com/wojciechkowalczyk11to-tech/nexus-omega-core
- Issues: Create GitHub issue
- Documentation: See `PROJECT_SUMMARY.md`

---

**Happy deploying! 🚀**
