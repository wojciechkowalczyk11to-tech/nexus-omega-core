# VM Go-Live Checklist (NexusOmegaCore)

## 1) Przygotowanie VM

- [ ] Zainstalowany Docker + Docker Compose (`docker compose version` działa)
- [ ] Otwarty port `8000` (API) i ewentualnie `5432/6379` tylko jeśli potrzebne z zewnątrz
- [ ] Ustawiona strefa czasu i NTP na VM

## 2) Kod i konfiguracja

```bash
git clone https://github.com/wojciechkowalczyk11to-tech/nexus-omega-core.git
cd nexus-omega-core
cp .env.example .env
```

- [ ] W `.env` ustawione: `TELEGRAM_BOT_TOKEN`, `JWT_SECRET_KEY`, `POSTGRES_PASSWORD`
- [ ] Zmienione domyślne kody: `DEMO_UNLOCK_CODE`, `BOOTSTRAP_ADMIN_CODE`
- [ ] Ustawiony min. 1 provider AI (`GEMINI_API_KEY` lub inny)

## 3) Pre-flight (musi przejść)

```bash
docker compose -f docker-compose.production.yml config -q
```

- [ ] Komenda kończy się bez błędu

## 4) Start produkcyjny

```bash
docker compose -f docker-compose.production.yml up --build -d
docker compose -f docker-compose.production.yml exec -T backend alembic upgrade head
```

- [ ] Wszystkie kontenery `Up`:

```bash
docker compose -f docker-compose.production.yml ps
```

## 5) Healthcheck i logi

```bash
python3 scripts/wait_for_backend_health.py --url http://localhost:8000/api/v1/health --timeout 240
curl http://localhost:8000/api/v1/health
```

- [ ] Odpowiedź health to `{"status":"healthy","database":"healthy","redis":"healthy"}`
- [ ] Brak krytycznych błędów w logach:

```bash
docker compose -f docker-compose.production.yml logs --tail=200 backend worker telegram_bot
```

## 6) Telegram test E2E

- [ ] `/start` działa
- [ ] `/unlock <DEMO_UNLOCK_CODE>` działa
- [ ] Zwykła wiadomość zwraca odpowiedź modelu

## 7) Operacyjne minimum po wdrożeniu

- [ ] Backup DB skonfigurowany (`scripts/backup_db.sh` + cron/systemd timer)
- [ ] Rotacja logów monitorowana (compose ma limity `max-size/max-file`)
- [ ] Procedura restore sprawdzona (`scripts/restore_db.sh`)

## 8) Rollback (awaryjnie)

```bash
docker compose -f docker-compose.production.yml down
# opcjonalnie powrót do poprzedniego commita/tagu
git checkout <POPRZEDNI_TAG_LUB_COMMIT>
docker compose -f docker-compose.production.yml up --build -d
```
