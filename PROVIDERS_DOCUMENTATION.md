# 📚 NexusOmegaCore — Dokumentacja Providerów AI

> Kompleksowy opis wszystkich providerów AI, modeli, funkcji, cennika i statusu operacyjnego.
>
> Ostatnia aktualizacja: 2026-02-21

---

## 📋 Spis treści

1. [Przegląd architektury providerów](#przegląd-architektury-providerów)
2. [Tabela zbiorcza providerów](#tabela-zbiorcza-providerów)
3. [Szczegółowy opis providerów](#szczegółowy-opis-providerów)
   - [Google Gemini](#1-google-gemini)
   - [DeepSeek](#2-deepseek)
   - [Groq](#3-groq)
   - [OpenRouter](#4-openrouter)
   - [xAI Grok](#5-xai-grok)
   - [OpenAI](#6-openai)
   - [Anthropic Claude](#7-anthropic-claude)
4. [System profili (ECO / SMART / DEEP)](#system-profili)
5. [SLM Router — routing kosztowy](#slm-router)
6. [Fallback chain — łańcuch awaryjny](#fallback-chain)
7. [RBAC — dostęp do providerów](#rbac--dostęp-do-providerów)
8. [Narzędzia (Tools)](#narzędzia-tools)
9. [Konfiguracja i zmienne środowiskowe](#konfiguracja)
10. [Status operacyjny i healthcheck](#status-operacyjny)
11. [FAQ](#faq)

---

## Przegląd architektury providerów

NexusOmegaCore wykorzystuje architekturę **multi-provider** z automatycznym routingiem i fallbackiem. Każdy provider AI implementuje wspólny interfejs `BaseProvider`, co zapewnia jednolity sposób wywoływania generacji tekstu niezależnie od dostawcy.

### Schemat przepływu zapytania

```
Użytkownik → Telegram Bot → Backend API → PolicyEngine (RBAC)
    → ModelRouter (klasyfikacja trudności)
    → ProviderFactory (tworzenie instancji providera)
    → Provider.generate() (wywołanie API)
    → ProviderResponse (ustandaryzowana odpowiedź)
    → UsageService (rozliczenie kosztów)
    → Odpowiedź do użytkownika
```

### Interfejs BaseProvider

Każdy provider implementuje następujące metody:

| Metoda | Opis |
|--------|------|
| `generate(messages, model, temperature, max_tokens)` | Generacja odpowiedzi z listy wiadomości |
| `get_model_for_profile(profile)` | Dobór modelu dla profilu (eco/smart/deep) |
| `calculate_cost(model, input_tokens, output_tokens)` | Kalkulacja kosztu w USD |
| `is_available()` | Sprawdzenie czy provider ma klucz API |
| `name` | Identyfikator providera (np. `"gemini"`) |
| `display_name` | Nazwa wyświetlana (np. `"Google Gemini"`) |

### Standardowa odpowiedź (ProviderResponse)

Każdy provider zwraca obiekt `ProviderResponse` ze standardowymi polami:

| Pole | Typ | Opis |
|------|-----|------|
| `content` | `str` | Wygenerowana treść odpowiedzi |
| `model` | `str` | Identyfikator użytego modelu |
| `input_tokens` | `int` | Liczba tokenów wejściowych |
| `output_tokens` | `int` | Liczba tokenów wyjściowych |
| `cost_usd` | `float` | Koszt zapytania w USD |
| `latency_ms` | `int` | Czas odpowiedzi w milisekundach |
| `finish_reason` | `str` | Powód zakończenia (np. `"stop"`) |
| `raw_response` | `dict` | Surowa odpowiedź z API providera |

---

## Tabela zbiorcza providerów

| # | Provider | Modele | Tier cenowy | Dostęp DEMO | Dostęp FULL | Klucz API |
|---|----------|--------|-------------|-------------|-------------|-----------|
| 1 | **Google Gemini** | gemini-2.0-flash-exp, gemini-2.0-flash-thinking-exp-1219, gemini-exp-1206 | Darmowy (free tier) | ✅ | ✅ | `GEMINI_API_KEY` |
| 2 | **DeepSeek** | deepseek-chat, deepseek-reasoner | Bardzo tani | ✅ (50/dzień) | ✅ | `DEEPSEEK_API_KEY` |
| 3 | **Groq** | llama-3.3-70b-versatile | Darmowy | ✅ | ✅ | `GROQ_API_KEY` |
| 4 | **OpenRouter** | llama-3.2-3b-instruct:free, llama-3.1-8b-instruct:free | Darmowy (free tier) | ✅ | ✅ | `OPENROUTER_API_KEY` |
| 5 | **xAI Grok** | grok-beta, grok-2-latest | Premium | ✅ (5/dzień) | ✅ | `XAI_API_KEY` |
| 6 | **OpenAI** | gpt-4o-mini, gpt-4o, gpt-4-turbo | Średni–Premium | ❌ | ✅ | `OPENAI_API_KEY` |
| 7 | **Anthropic Claude** | claude-3-5-haiku, claude-3-5-sonnet, claude-3-opus | Średni–Premium | ❌ | ✅ | `ANTHROPIC_API_KEY` |

---

## Szczegółowy opis providerów

### 1. Google Gemini

**Klasa:** `GeminiProvider`
**Plik:** `backend/app/providers/gemini_provider.py`
**Biblioteka:** `google-generativeai`

#### Opis

Google Gemini to główny provider do zadań ekonomicznych (profil ECO). Oferuje darmowe modele eksperymentalne z dużym oknem kontekstowym (do 2M tokenów). Provider konwertuje wiadomości z formatu OpenAI na format Gemini (mapowanie ról: `system` → `user [System]`, `assistant` → `model`).

#### Modele

| Model | Profil | Okno kontekstu | Koszt input / 1M tokenów | Koszt output / 1M tokenów | Uwagi |
|-------|--------|----------------|---------------------------|----------------------------|-------|
| `gemini-2.0-flash-exp` | ECO | 1 000 000 | $0.00 | $0.00 | Darmowy, szybki, eksperymentalny |
| `gemini-2.0-flash-thinking-exp-1219` | SMART | 32 000 | $0.00 | $0.00 | Darmowy, z reasoning |
| `gemini-exp-1206` | DEEP | 2 000 000 | $0.00 | $0.00 | Darmowy, największe okno kontekstu |
| `gemini-1.5-flash` | (dodatkowy) | 1 000 000 | $0.075 | $0.30 | Płatny, produkcyjny |
| `gemini-1.5-pro` | (dodatkowy) | 2 000 000 | $1.25 | $5.00 | Płatny, najwyższa jakość |

#### Funkcje

- ✅ Generacja tekstu (chat completions)
- ✅ Konwersja wiadomości systemowych na format Gemini
- ✅ Fallback estimacja tokenów (gdy brak usage_metadata)
- ✅ Asynchroniczne wywołania przez `run_in_executor`
- ✅ Konfigurowalny temperature i max_tokens
- ❌ Brak natywnego wsparcia dla system prompt (emulowany)

#### Specyfika implementacji

- Używa `genai.GenerativeModel` z synchronicznym `generate_content` opakowanym w `loop.run_in_executor` dla async kompatybilności
- Rola `system` jest mapowana na `user` z prefiksem `[System]`
- Rola `assistant` jest mapowana na `model`

---

### 2. DeepSeek

**Klasa:** `DeepSeekProvider`
**Plik:** `backend/app/providers/deepseek_provider.py`
**Biblioteka:** `openai` (kompatybilne API)

#### Opis

DeepSeek to chiński provider AI oferujący modele o bardzo niskim koszcie. Wykorzystuje API kompatybilne z OpenAI (`base_url: https://api.deepseek.com`). Jest głównym providerem dla profilu SMART i DEEP ze względu na doskonały stosunek jakości do ceny.

#### Modele

| Model | Profil | Okno kontekstu | Koszt input / 1M tokenów | Koszt output / 1M tokenów | Uwagi |
|-------|--------|----------------|---------------------------|----------------------------|-------|
| `deepseek-chat` | ECO | 64 000 | $0.14 | $0.28 | Szybki, ekonomiczny |
| `deepseek-reasoner` | SMART, DEEP | 64 000 | $0.55 | $2.19 | Z reasoning (chain-of-thought) |

#### Funkcje

- ✅ Pełna kompatybilność z OpenAI API
- ✅ Chat completions
- ✅ Reasoning (deepseek-reasoner)
- ✅ System prompts
- ✅ Precyzyjne usage tracking (prompt_tokens, completion_tokens)

---

### 3. Groq

**Klasa:** `GroqProvider`
**Plik:** `backend/app/providers/groq_provider.py`
**Biblioteka:** `openai` (kompatybilne API)

#### Opis

Groq to provider specjalizujący się w ultra-szybkim inferenzie na dedykowanym hardware (LPU™). Oferuje darmowy tier z modelami Llama. Wykorzystuje API kompatybilne z OpenAI (`base_url: https://api.groq.com/openai/v1`).

#### Modele

| Model | Profil | Okno kontekstu | Koszt input / 1M tokenów | Koszt output / 1M tokenów | Uwagi |
|-------|--------|----------------|---------------------------|----------------------------|-------|
| `llama-3.3-70b-versatile` | ECO, SMART, DEEP | 128 000 | $0.00 | $0.00 | Darmowy, Llama 3.3 70B |
| `llama-3.1-70b-versatile` | (dodatkowy) | 128 000 | $0.00 | $0.00 | Darmowy, Llama 3.1 70B |

#### Funkcje

- ✅ Ultra-szybki inference (LPU hardware)
- ✅ Darmowy tier
- ✅ Pełna kompatybilność z OpenAI API
- ✅ Chat completions
- ✅ System prompts
- ⚠️ Rate limiting na darmowym tierze

#### Specyfika implementacji

- Metoda `calculate_cost()` zawsze zwraca `0.0` (darmowy tier)
- Idealny jako fallback provider w łańcuchu awaryjnym

---

### 4. OpenRouter

**Klasa:** `OpenRouterProvider`
**Plik:** `backend/app/providers/openrouter_provider.py`
**Biblioteka:** `openai` (kompatybilne API)

#### Opis

OpenRouter to agregator modeli AI oferujący dostęp do wielu modeli przez jedno API. NexusOmegaCore używa wyłącznie darmowych modeli z free tier. Wykorzystuje API kompatybilne z OpenAI (`base_url: https://openrouter.ai/api/v1`).

#### Modele

| Model | Profil | Okno kontekstu | Koszt input / 1M tokenów | Koszt output / 1M tokenów | Uwagi |
|-------|--------|----------------|---------------------------|----------------------------|-------|
| `meta-llama/llama-3.2-3b-instruct:free` | ECO | 64 000 | $0.00 | $0.00 | Darmowy, mały model |
| `meta-llama/llama-3.1-8b-instruct:free` | SMART, DEEP | 64 000 | $0.00 | $0.00 | Darmowy, średni model |

#### Funkcje

- ✅ Dostęp do wielu modeli przez jedno API
- ✅ Darmowy tier
- ✅ Pełna kompatybilność z OpenAI API
- ✅ Chat completions
- ⚠️ Rate limiting na darmowym tierze
- ⚠️ Ograniczona jakość na darmowych modelach

---

### 5. xAI Grok

**Klasa:** `GrokProvider`
**Plik:** `backend/app/providers/grok_provider.py`
**Biblioteka:** `openai` (kompatybilne API)

#### Opis

Grok to model AI od xAI (firmy Elona Muska). Oferuje zaawansowane możliwości konwersacyjne. Wykorzystuje API kompatybilne z OpenAI (`base_url: https://api.x.ai/v1`). Jest providerem premium z limitem 5 zapytań dziennie dla użytkowników DEMO.

#### Modele

| Model | Profil | Okno kontekstu | Koszt input / 1M tokenów | Koszt output / 1M tokenów | Uwagi |
|-------|--------|----------------|---------------------------|----------------------------|-------|
| `grok-beta` | ECO, SMART, DEEP | 128 000 | $5.00 | $15.00 | Premium, beta |
| `grok-2-latest` | (dodatkowy) | 128 000 | $5.00 | $15.00 | Premium, stabilny |

#### Funkcje

- ✅ Zaawansowany reasoning
- ✅ Pełna kompatybilność z OpenAI API
- ✅ Chat completions
- ✅ System prompts
- ⚠️ Limit 5/dzień dla DEMO użytkowników
- ⚠️ Wysoki koszt ($5-15 / 1M tokenów)

---

### 6. OpenAI

**Klasa:** `OpenAIProvider`
**Plik:** `backend/app/providers/openai_provider.py`
**Biblioteka:** `openai`

#### Opis

OpenAI GPT to flagowy provider dla zadań wymagających najwyższej jakości. Dostępny wyłącznie dla użytkowników z rolą FULL_ACCESS lub ADMIN. Oferuje modele od ekonomicznego GPT-4o-mini po zaawansowany GPT-4-turbo.

#### Modele

| Model | Profil | Okno kontekstu | Koszt input / 1M tokenów | Koszt output / 1M tokenów | Uwagi |
|-------|--------|----------------|---------------------------|----------------------------|-------|
| `gpt-4o-mini` | ECO | 128 000 | $0.15 | $0.60 | Ekonomiczny, szybki |
| `gpt-4o` | SMART, DEEP | 128 000 | $2.50 | $10.00 | Balans jakość/koszt |
| `gpt-4-turbo` | (dodatkowy) | 128 000 | $10.00 | $30.00 | Najwyższa jakość |

#### Funkcje

- ✅ Natywne API OpenAI (AsyncOpenAI)
- ✅ Chat completions
- ✅ System prompts
- ✅ Function calling
- ✅ Precyzyjne usage tracking
- ✅ Streaming (obsługiwane przez bibliotekę)
- ❌ Dostęp tylko dla FULL_ACCESS / ADMIN

---

### 7. Anthropic Claude

**Klasa:** `ClaudeProvider`
**Plik:** `backend/app/providers/claude_provider.py`
**Biblioteka:** `anthropic`

#### Opis

Anthropic Claude to provider premium z dedykowanym API. Specjalizuje się w zadaniach analitycznych, długich konwersacjach i kodowaniu. Dostępny wyłącznie dla użytkowników FULL_ACCESS / ADMIN. Implementacja wyodrębnia system prompt z listy wiadomości i przekazuje go osobno (specyfika API Anthropic).

#### Modele

| Model | Profil | Okno kontekstu | Koszt input / 1M tokenów | Koszt output / 1M tokenów | Uwagi |
|-------|--------|----------------|---------------------------|----------------------------|-------|
| `claude-3-5-haiku-20241022` | ECO | 200 000 | $0.80 | $4.00 | Szybki, ekonomiczny |
| `claude-3-5-sonnet-20241022` | SMART, DEEP | 200 000 | $3.00 | $15.00 | Balans jakość/koszt |
| `claude-3-opus-20240229` | (dodatkowy) | 200 000 | $15.00 | $75.00 | Najwyższa jakość |

#### Funkcje

- ✅ Dedykowane API Anthropic (AsyncAnthropic)
- ✅ Separacja system prompt (natywna obsługa)
- ✅ Chat completions
- ✅ Duże okno kontekstu (200K tokenów)
- ✅ Function calling (tool use)
- ✅ Precyzyjne usage tracking
- ❌ Dostęp tylko dla FULL_ACCESS / ADMIN

#### Specyfika implementacji

- System message jest wyodrębniany z listy wiadomości i przekazywany jako osobny parametr `system` w żądaniu API
- Pozostałe wiadomości (user/assistant) są przekazywane normalnie

---

## System profili

NexusOmegaCore implementuje trzy profile jakościowe, które wpływają na dobór modelu u każdego providera:

### ECO (ekonomiczny)

- **Cel:** Minimalizacja kosztów
- **Użycie:** Proste pytania, szybkie odpowiedzi
- **Klasyfikacja:** Zapytania łatwe (DifficultyLevel.EASY)
- **Łańcuch providerów:** Gemini → Groq → DeepSeek

| Provider | Model ECO |
|----------|-----------|
| Gemini | `gemini-2.0-flash-exp` |
| DeepSeek | `deepseek-chat` |
| Groq | `llama-3.3-70b-versatile` |
| OpenRouter | `meta-llama/llama-3.2-3b-instruct:free` |
| Grok | `grok-beta` |
| OpenAI | `gpt-4o-mini` |
| Claude | `claude-3-5-haiku-20241022` |

### SMART (zbalansowany)

- **Cel:** Balans między jakością a kosztem
- **Użycie:** Pytania średniej trudności, wyjaśnienia, kod
- **Klasyfikacja:** Zapytania średnie (DifficultyLevel.MEDIUM)
- **Łańcuch providerów:** DeepSeek → Gemini → Groq

| Provider | Model SMART |
|----------|------------|
| Gemini | `gemini-2.0-flash-thinking-exp-1219` |
| DeepSeek | `deepseek-reasoner` |
| Groq | `llama-3.3-70b-versatile` |
| OpenRouter | `meta-llama/llama-3.1-8b-instruct:free` |
| Grok | `grok-beta` |
| OpenAI | `gpt-4o` |
| Claude | `claude-3-5-sonnet-20241022` |

### DEEP (premium)

- **Cel:** Maksymalna jakość odpowiedzi
- **Użycie:** Złożone analizy, architektura, optymalizacja
- **Klasyfikacja:** Zapytania trudne (DifficultyLevel.HARD)
- **Łańcuch providerów:** DeepSeek → Gemini → OpenAI → Claude
- **Dostęp:** Tylko FULL_ACCESS i ADMIN

| Provider | Model DEEP |
|----------|-----------|
| Gemini | `gemini-exp-1206` |
| DeepSeek | `deepseek-reasoner` |
| Groq | `llama-3.3-70b-versatile` |
| OpenRouter | `meta-llama/llama-3.1-8b-instruct:free` |
| Grok | `grok-beta` |
| OpenAI | `gpt-4o` |
| Claude | `claude-3-5-sonnet-20241022` |

---

## SLM Router

System **SLM-first Cost-Aware Router** preferuje małe, tanie modele i eskaluje tylko gdy jest to konieczne.

### Tiery modeli

| Tier | Koszt ~/ 1M tokenów | Modele | Prędkość |
|------|---------------------|--------|----------|
| **ULTRA_CHEAP** | ~$0.10 | Groq Llama 3.1 8B, Gemini Flash | ⚡⚡⚡ |
| **CHEAP** | ~$0.50 | DeepSeek Chat, Gemini 1.5 Pro | ⚡⚡ |
| **BALANCED** | ~$2.00 | GPT-4o-mini, Claude Sonnet | ⚡ |
| **PREMIUM** | ~$10.00+ | GPT-4-turbo, Claude Opus | 🐌 |

### Logika eskalacji

1. Rozpoczyna od **ULTRA_CHEAP** tier
2. Jeśli `cost_preference = LOW` → obniża tier o 1
3. Jeśli `cost_preference = QUALITY` → podnosi tier o 1
4. Jeśli brak pasujących modeli → eskaluje do następnego tieru
5. Heurystyka `should_escalate()` porównuje `task_complexity_score` z `model_capability`

### Preferencje kosztowe użytkownika

| Preferencja | Opis | Domyślny tier dla "moderate" |
|-------------|------|------------------------------|
| `LOW` | Minimalizuj koszt | ULTRA_CHEAP |
| `BALANCED` | Balans | CHEAP |
| `QUALITY` | Priorytet jakość | BALANCED |

---

## Fallback chain

System automatycznie przechodzi do następnego providera w łańcuchu jeśli bieżący zawiedzie.

### Łańcuchy awaryjne

```
ECO:   gemini → groq → deepseek
SMART: deepseek → gemini → groq
DEEP:  deepseek → gemini → openai → claude
```

### Mechanizm

1. `ProviderFactory.generate_with_fallback()` iteruje po łańcuchu
2. Dla każdego providera: tworzy instancję → dobiera model → generuje
3. Jeśli `ProviderError` → loguje ostrzeżenie → próbuje następnego
4. Jeśli wszystkie zawiodą → rzuca `AllProvidersFailedError`
5. Zwraca tuple: `(ProviderResponse, provider_name, fallback_used)`

### Filtrowanie łańcucha po RBAC

Łańcuch jest filtrowany na podstawie roli użytkownika. Np. użytkownik DEMO z profilem DEEP dostanie łańcuch `deepseek → gemini` (bez `openai` i `claude`).

---

## RBAC — dostęp do providerów

### Macierz dostępu

| Provider | DEMO | FULL_ACCESS | ADMIN |
|----------|------|-------------|-------|
| Gemini | ✅ | ✅ | ✅ |
| DeepSeek | ✅ (50/dzień) | ✅ | ✅ |
| Groq | ✅ | ✅ | ✅ |
| OpenRouter | ✅ | ✅ | ✅ |
| Grok | ✅ (5/dzień) | ✅ | ✅ |
| OpenAI | ❌ | ✅ | ✅ |
| Claude | ❌ | ✅ | ✅ |

### Limity dzienne (DEMO)

| Zasób | Limit dzienny |
|-------|---------------|
| Grok calls | 5 |
| Web search calls | 5 |
| Smart credits | 20 |
| DeepSeek calls | 50 |

### Budżet dzienny (USD)

| Rola | Budżet |
|------|--------|
| DEMO | $0.00 |
| FULL_ACCESS | $5.00 |
| ADMIN | Bez limitu |

### Smart Credits

Kalkulacja na podstawie łącznej liczby tokenów:

| Tokeny | Kredyty |
|--------|---------|
| ≤ 500 | 1 |
| ≤ 2000 | 2 |
| > 2000 | 4 |

---

## Narzędzia (Tools)

Oprócz providerów AI, system oferuje zestaw narzędzi zintegrowanych z ReAct Orchestratorem:

### Dostępne narzędzia

| Narzędzie | Opis | Wymagany klucz API |
|-----------|------|---------------------|
| **Web Search** | Wyszukiwanie w internecie (Brave Search API) | `BRAVE_SEARCH_API_KEY` |
| **Vertex AI Search** | Wyszukiwanie w bazie wiedzy z cytatami | `VERTEX_PROJECT_ID`, `VERTEX_SEARCH_DATASTORE_ID` |
| **RAG Search** | Wyszukiwanie semantyczne w dokumentach użytkownika | — (wbudowane) |
| **Calculate** | Obliczenia matematyczne (safe eval) | — |
| **Get DateTime** | Pobranie aktualnej daty i czasu | — |
| **Memory Read/Write** | Odczyt/zapis do pamięci absolutnej użytkownika | — |
| **GitHub Devin** | Klonowanie repo, edycja kodu, tworzenie PR | `GITHUB_TOKEN` |

### ReAct Orchestrator

System **ReAct (Reason-Act-Observe-Think)** zarządza pętlą narzędziową:

1. **REASON** — LLM analizuje zapytanie i decyduje o użyciu narzędzia
2. **ACT** — Wykonanie narzędzia lub generacja odpowiedzi
3. **OBSERVE** — Analiza wyniku narzędzia
4. **THINK** — Self-correction: czy wynik jest poprawny?
5. **RESPOND** — Finalna odpowiedź do użytkownika

- Maks. 6 iteracji pętli ReAct
- Maks. 2 self-corrections na iterację

### Token Budget Manager

System inteligentnego zarządzania budżetem tokenów:

- **Priorytetyzacja:** System prompt (90) > Bieżące zapytanie (100) > Pamięć (65) > Historia (30-40) > Snapshot (10)
- **Smart truncation:** Zachowuje pierwszy i ostatni akapit, skraca środek
- **Rezerwa na odpowiedź:** 15% kontekstu
- **Margines bezpieczeństwa:** 5% kontekstu

---

## Konfiguracja

### Zmienne środowiskowe providerów

```env
# Wymagany (minimum 1 provider)
GEMINI_API_KEY=         # Google Gemini — główny darmowy provider
DEEPSEEK_API_KEY=       # DeepSeek — tani, dobry reasoning
GROQ_API_KEY=           # Groq — darmowy, ultra-szybki

# Opcjonalne (free tier)
OPENROUTER_API_KEY=     # OpenRouter — agregator darmowych modeli

# Opcjonalne (premium)
XAI_API_KEY=            # xAI Grok — premium conversational AI
OPENAI_API_KEY=         # OpenAI GPT — tylko FULL_ACCESS/ADMIN
ANTHROPIC_API_KEY=      # Anthropic Claude — tylko FULL_ACCESS/ADMIN
```

### Konfiguracja policy (JSON)

```env
PROVIDER_POLICY_JSON={"default":{"providers":{"gemini":{"enabled":true},"deepseek":{"enabled":true},"groq":{"enabled":true}}}}
```

### Aliasy providerów

System automatycznie normalizuje nazwy providerów:

| Alias | Mapowany na |
|-------|-------------|
| `xai`, `x.ai` | `grok` |
| `google` | `gemini` |
| `anthropic` | `claude` |
| `llama` | `groq` |

---

## Status operacyjny

### Healthcheck

```bash
curl http://localhost:8000/api/v1/health
```

Oczekiwana odpowiedź:
```json
{
  "status": "healthy",
  "database": "healthy",
  "redis": "healthy"
}
```

### Sprawdzenie dostępnych providerów

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/chat/providers
```

### Monitoring

- **Logi:** JSON format z request tracing (`LOG_JSON=true`)
- **Usage tracking:** Tabela `usage_ledger` z pełnym rozliczeniem kosztów
- **Tool counters:** Tabela `tool_counters` ze zliczaniem użycia dziennego
- **Audit log:** Tabela `audit_logs` z akcjami administracyjnymi
- **Agent traces:** Tabela `agent_traces` z pełnym śladem rozumowania ReAct

### Wdrożenie na VM

1. Zainstaluj Docker i Docker Compose na VM
2. Sklonuj repozytorium: `git clone <repo_url>`
3. Skopiuj `.env.example` do `.env` i wypełnij klucze API
4. Uruchom: `docker compose -f docker-compose.production.yml up -d`
5. Sprawdź healthcheck: `curl http://localhost:8000/api/v1/health`
6. Zweryfikuj bota w Telegram: `/start`

Szczegółowe instrukcje w [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md).

---

## FAQ

### Który provider wybrać jako minimum?

**Gemini** — darmowy, duże okno kontekstu, dobra jakość. Wystarczy jako jedyny provider.

### Ile kosztuje typowe zapytanie?

| Profil | Provider | ~Koszt na zapytanie (1K in / 500 out) |
|--------|----------|---------------------------------------|
| ECO | Gemini | $0.00 (darmowy) |
| ECO | Groq | $0.00 (darmowy) |
| SMART | DeepSeek | ~$0.001 |
| SMART | OpenAI (gpt-4o) | ~$0.008 |
| DEEP | Claude Sonnet | ~$0.011 |
| DEEP | Claude Opus | ~$0.053 |

### Jak dodać nowego providera?

1. Utwórz plik `backend/app/providers/<name>_provider.py`
2. Zaimplementuj klasę dziedziczącą z `BaseProvider`
3. Zarejestruj w `ProviderFactory.PROVIDERS` i `_get_api_key()`
4. Dodaj klucz API do `Settings` w `core/config.py`
5. Dodaj do `.env.example`
6. Zaktualizuj `PolicyEngine.PROVIDER_ACCESS` dla ról

### Co się stanie gdy provider zawiedzie?

System automatycznie przejdzie do następnego providera w łańcuchu awaryjnym. Jeśli wszystkie zawiodą, użytkownik otrzyma komunikat: *"Wszystkie providery AI zawiodły. Spróbuj ponownie później."*

### Jak działa klasyfikacja trudności?

Wielosygnałowy scoring:
1. **Słowa kluczowe** (PL + EN) — waga 40% (hard) / 30% (medium)
2. **Złożoność strukturalna** — waga 50% (długość, bloki kodu, listy)
3. **Detekcja intencji** — bonus 30% (analityczne, kod)
4. **Scoring:** ≥0.5 = HARD, ≥0.15 = MEDIUM, <0.15 = EASY

---

*Dokument wygenerowany automatycznie na podstawie kodu źródłowego NexusOmegaCore.*
