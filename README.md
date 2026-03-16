# TokenRouter

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/byok-router.svg)](https://pypi.org/project/byok-router/)

**Open-source AI model routing platform** — BYOK (Bring Your Own Key), auto-select the optimal model based on prompt content. Self-hosted, zero data leakage.

TokenRouter analyzes your prompt, classifies the task type (coding, math, translation, etc.), estimates complexity, and routes to the best model — **saving 60-80% on API costs** without sacrificing quality.

## Why TokenRouter?

| Without TokenRouter | With TokenRouter |
|---|---|
| Every request → expensive model (GPT-5.2 / Claude Opus) | Simple questions → cheap model, complex ones → premium |
| Manual model switching in code | One API key, automatic routing |
| No visibility into costs | Real-time cost tracking + savings dashboard |
| Single provider failure = downtime | Automatic fallback chain |

## Features

- 🧠 **Smart Routing** — L1 regex + L2 model classifier, auto-selects optimal model per request
- 🔑 **Unified API Key** — One `tr-xxx` key, routes to OpenAI/Anthropic/Google/DeepSeek behind the scenes
- 💰 **Cost Tracking** — Every request logged with actual cost vs baseline, see exactly how much you save
- 📊 **Built-in Dashboard** — Dark-themed web UI with cost charts, model distribution, request logs
- 🔄 **Fallback Chain** — Automatic retry with fallback models on failure (429/5xx)
- 🌐 **OpenAI-Compatible** — Drop-in replacement for any OpenAI SDK
- 🏠 **Self-Hosted** — Your keys never leave your server
- ⚡ **3 Strategies** — `cheapest`, `best`, `balanced` (quality/cost ratio)

## Quick Start

### Install

```bash
pip install byok-router
```

### Option 1: Self-Hosted Platform (Recommended)

Set up once, use everywhere:

```bash
# 1. Create a unified API key
tokenrouter keys create --name "my-project" --strategy balanced

# Output: Created key tr-a1b2c3d4... (id: 1)

# 2. Add your provider keys
tokenrouter keys add-provider 1 openai sk-...
tokenrouter keys add-provider 1 anthropic sk-ant-...
tokenrouter keys add-provider 1 deepseek sk-...

# 3. Start the server
tokenrouter serve --port 8000

# Dashboard: http://localhost:8000
# API: http://localhost:8000/v1
```

Now use your `tr-xxx` key with **any** OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tr-a1b2c3d4..."  # your unified key
)

response = client.chat.completions.create(
    model="auto",  # TokenRouter picks the best model
    messages=[{"role": "user", "content": "Write a Python quicksort"}]
)

print(response.model)  # "deepseek-chat" (cheapest for coding)
# Check headers for cost info:
# X-TokenRouter-Cost: 0.00032
# X-TokenRouter-Saved: 0.04500
# X-TokenRouter-Saved-Pct: 99.3
```

### Option 2: Python SDK (Direct)

```python
from tokenrouter import TokenRouter

router = TokenRouter(
    keys={
        "openai": "sk-...",
        "anthropic": "sk-ant-...",
        "deepseek": "sk-...",
    },
    strategy="balanced",
)

response = router.chat([
    {"role": "user", "content": "Write a Python quicksort function"}
])
print(response.model_used)  # "deepseek-chat"
print(response.choices[0].message["content"])

# Streaming
for chunk in router.chat_stream([
    {"role": "user", "content": "Explain quantum computing"}
]):
    delta = chunk.choices[0].get("delta", {})
    print(delta.get("content", ""), end="")
```

### Option 3: YAML Config

```yaml
# config.yaml
admin_token: ${TOKENROUTER_ADMIN_TOKEN}
database: ~/.tokenrouter/keys.db
baseline_model: claude-opus-4  # cost comparison baseline

# Legacy mode: direct keys (single user, no tr-key needed)
keys:
  openai: ${OPENAI_API_KEY}
  anthropic: ${ANTHROPIC_API_KEY}
  deepseek: ${DEEPSEEK_API_KEY}

strategy: balanced

rules:
  - task: coding
    model: deepseek-chat
  - task: chinese_language
    model: qwen-max

exclude_models:
  - claude-opus-4

proxy:
  host: 0.0.0.0
  port: 8000
  cors: true
```

```bash
tokenrouter serve --config config.yaml
```

## Dashboard

Access the built-in dashboard at `http://localhost:8000` after starting the server.

**What you see:**
- 📈 Total requests, cost, savings, and savings percentage
- 📉 Daily cost vs baseline line chart
- 🍩 Model usage distribution
- 📋 Recent request logs with task type, model, cost, and latency
- 🔑 Key management (create, view, delete)

## Cost Tracking

Every request is tracked with:

| Field | Description |
|-------|-------------|
| `actual_cost` | What you actually paid |
| `baseline_cost` | What it would cost with Claude Opus / GPT-5.2 |
| `saved` | `baseline_cost - actual_cost` |
| `saved_pct` | Savings percentage |

```bash
# View usage from CLI
tokenrouter usage 1  # key id

# API endpoint
curl http://localhost:8000/v1/usage/1/savings \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

Example savings report:
```json
{
  "total_requests": 1247,
  "total_cost": 1.83,
  "baseline_cost": 12.45,
  "saved": 10.62,
  "saved_pct": 85.3,
  "period": "month"
}
```

## API Key Management

### CLI

```bash
tokenrouter keys create --name "production" --strategy balanced
tokenrouter keys list
tokenrouter keys add-provider 1 openai sk-...
tokenrouter keys add-provider 1 anthropic sk-ant-...
tokenrouter keys delete 1
```

### REST API

```bash
# Create key (requires admin token)
curl -X POST http://localhost:8000/v1/keys \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-key", "strategy": "balanced"}'

# List keys
curl http://localhost:8000/v1/keys \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Add provider key
curl -X POST http://localhost:8000/v1/keys/1/providers \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "api_key": "sk-..."}'

# Get savings report
curl http://localhost:8000/v1/usage/1/savings \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

## How Routing Works

```
User Request → L1 Classifier → L2 Classifier (if needed) → Routing Table → Strategy Selection → Provider
                  (<1ms)           (~200ms)                  (24 combos)     (cheapest/best/     → Fallback
                                                                              balanced)             Chain
```

1. **L1 Classifier** — Regex heuristic scores 8 task categories (coding, math, translation, creative writing, summarization, reasoning, QA, Chinese)
2. **L2 Classifier** — When L1 confidence < 0.7, uses a cheap model to refine classification
3. **Routing Table** — Maps `task_type:complexity` to candidate models
4. **Strategy Selection** — Picks from candidates using capability matrix (each model scored 1-5 on 8 dimensions)
5. **Fallback Chain** — If primary model returns 429/5xx, auto-retries with next candidate

## Supported Models

| Provider | Models | Best For | Cost (per 1M tokens) |
|----------|--------|----------|---------------------|
| **OpenAI** | GPT-5.2, GPT-5 Mini | General, coding | $1.75-14 / $0.25-2 |
| **Anthropic** | Opus 4/4.5, Sonnet 4, Haiku 4.5 | Coding, writing | $1-15 / $5-75 |
| **Google** | Gemini 3 Flash, 2.5 Flash/Pro | Quick tasks | $0.30-1.25 / $2.50-10 |
| **DeepSeek** | V3.2, R1 | Coding, math | $0.28 / $0.42 |
| **Moonshot** | Kimi K2.5 | Chinese, coding | $0.60 / $2.50 |
| **Qwen** | Turbo, Plus, Max | Chinese tasks | $0.05-1.60 / $0.20-6.40 |
| **Zhipu** | GLM-4 Plus | Chinese, QA | $0.50 / $1.50 |

## Use with OpenClaw

TokenRouter is designed to work seamlessly with [OpenClaw](https://github.com/openclaw/openclaw). Set up your `tr-xxx` key and point OpenClaw to your TokenRouter instance:

```yaml
# OpenClaw config
model:
  provider: openai-compatible
  baseUrl: http://localhost:8000/v1
  apiKey: tr-a1b2c3d4...
  model: auto
```

## Development

```bash
git clone https://github.com/aplomb2/TokenRouter.git
cd TokenRouter
pip install -e ".[dev]"
pytest  # 182 tests
```

## License

MIT — see [LICENSE](LICENSE).
