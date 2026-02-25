# AI Models Reference

## Supported Providers & Models

### Google Gemini

| Profile | Model ID | Input ($/1M) | Output ($/1M) | Context | Notes |
|---------|----------|-------------|---------------|---------|-------|
| ECO | gemini-2.5-flash-preview-05-20 | $0.15 | $0.60 | 1M | Fast, cost-effective |
| SMART | gemini-2.5-flash-preview-05-20 | $0.15 | $0.60 | 1M | With thinking: $3.50/1M output |
| DEEP | gemini-2.5-pro-preview-05-06 | $1.25 | $10.00 | 1M | Best reasoning quality |

### xAI Grok

| Profile | Model ID | Input ($/1M) | Output ($/1M) | Notes |
|---------|----------|-------------|---------------|-------|
| ECO | grok-3-mini-fast | $0.30 | $0.50 | Fastest Grok model |
| SMART | grok-3-fast | $5.00 | $15.00 | Fast reasoning |
| DEEP | grok-3 | $3.00 | $15.00 | Full capabilities |

### OpenAI

| Profile | Model ID | Input ($/1M) | Output ($/1M) | Context | Notes |
|---------|----------|-------------|---------------|---------|-------|
| ECO | gpt-4o-mini | $0.15 | $0.60 | 128K | Best value |
| SMART | gpt-4o | $2.50 | $10.00 | 128K | Flagship |
| DEEP | o3-mini | $1.10 | $4.40 | 200K | Advanced reasoning |

### Anthropic Claude

| Profile | Model ID | Input ($/1M) | Output ($/1M) | Context | Notes |
|---------|----------|-------------|---------------|---------|-------|
| ECO | claude-3-5-haiku-20241022 | $0.80 | $4.00 | 200K | Fast, efficient |
| SMART | claude-sonnet-4-20250514 | $3.00 | $15.00 | 200K | Balanced |
| DEEP | claude-opus-4-20250918 | $15.00 | $75.00 | 200K | 32K output tokens |

### DeepSeek

| Profile | Model ID | Input ($/1M) | Output ($/1M) | Context | Notes |
|---------|----------|-------------|---------------|---------|-------|
| ECO | deepseek-chat | $0.14 | $0.28 | 64K | Ultra-cheap |
| SMART | deepseek-chat | $0.14 | $0.28 | 64K | V3 model |
| DEEP | deepseek-reasoner | $0.55 | $2.19 | 64K | R1 reasoning |

### Groq (Ultra-Fast Inference)

| Model ID | Input ($/1M) | Output ($/1M) | Context | Notes |
|----------|-------------|---------------|---------|-------|
| llama-3.1-8b-instant | $0.05 | $0.08 | 8K | Fastest SLM option |

## SLM Router Tiers

| Tier | Models | Typical Cost | Use Case |
|------|--------|-------------|----------|
| ULTRA_CHEAP | Groq Llama 3.1 8B, Gemini 2.5 Flash | $0.05-$0.60/1M | Simple Q&A, translations |
| CHEAP | DeepSeek V3, Gemini 2.5 Pro | $0.14-$10.00/1M | Moderate analysis, summaries |
| BALANCED | GPT-4o-mini, Claude Sonnet 4 | $0.15-$15.00/1M | Complex reasoning, coding |
| PREMIUM | o3-mini, Claude Opus 4 | $1.10-$75.00/1M | Expert analysis, research |

## Last Updated

2026-02-25
