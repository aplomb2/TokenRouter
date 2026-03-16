# TokenRouter

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/tokenrouter.svg)](https://pypi.org/project/tokenrouter/)

**Lightweight AI model intelligent routing SDK** — BYOK (Bring Your Own Key), auto-select the optimal model based on prompt content. Local deployment, zero data leakage.

TokenRouter analyzes your prompt, classifies the task type (coding, math, translation, etc.), estimates complexity, and routes to the best model using a capability matrix and your preferred strategy.

## Features

- **Smart Routing** — Automatically selects the optimal model based on task type, complexity, and strategy
- **BYOK** — Use your own API keys. No middleman, no data leakage
- **Multi-Provider** — OpenAI, Anthropic, Google, DeepSeek, Moonshot, Qwen, Zhipu
- **3 Strategies** — `cheapest`, `best`, `balanced` (quality/cost ratio)
- **Fallback Chain** — Automatic retry with fallback models on failure (429, 500, 502, 503, 504)
- **Two-Tier Classifier** — L1 (regex heuristic, instant) + L2 (cheap model, when L1 confidence < 0.7)
- **OpenAI-Compatible Proxy** — Drop-in replacement for any OpenAI SDK
- **Async-First** — Built on httpx async, with sync wrappers for convenience
- **Zero Core Dependencies** — Core routing uses only stdlib; httpx for API calls; FastAPI optional for proxy
- **100% Type Hints** — Full type annotations throughout

## Installation

```bash
# Core (routing + providers)
pip install tokenrouter

# With proxy server
pip install tokenrouter[proxy]

# With YAML config support
pip install tokenrouter[yaml]

# Everything
pip install tokenrouter[all]
```

## Quick Start

### 1. Python SDK

```python
from tokenrouter import TokenRouter

router = TokenRouter(
    keys={
        "openai": "sk-...",
        "anthropic": "sk-ant-...",
        "deepseek": "sk-...",
    },
    strategy="balanced",  # cheapest | best | balanced
)

# Auto-route — TokenRouter picks the best model
response = router.chat([
    {"role": "user", "content": "Write a Python quicksort function"}
])
print(response.model_used)     # e.g. "deepseek-chat"
print(response.choices[0].message["content"])

# Streaming
for chunk in router.chat_stream([
    {"role": "user", "content": "Explain quantum computing"}
]):
    delta = chunk.choices[0].get("delta", {})
    print(delta.get("content", ""), end="")

# Classify only (no API call)
result = router.classify([
    {"role": "user", "content": "Translate to French: Hello"}
])
print(result.task_type)       # "translation"
print(result.selected_model)  # ModelConfig for the optimal model
```

### 2. Async API

```python
import asyncio
from tokenrouter import TokenRouter

router = TokenRouter(keys={"openai": "sk-..."})

async def main():
    response = await router.achat([
        {"role": "user", "content": "What is 2+2?"}
    ])
    print(response.model_used)

    async for chunk in router.achat_stream([
        {"role": "user", "content": "Write a haiku"}
    ]):
        delta = chunk.choices[0].get("delta", {})
        print(delta.get("content", ""), end="")

asyncio.run(main())
```

### 3. YAML Config

```yaml
# config.yaml
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
```

```python
router = TokenRouter.from_config("config.yaml")
```

### 4. OpenAI-Compatible Proxy

Start the proxy server:

```bash
tokenrouter serve --config config.yaml --port 8000
```

Then use with **any** OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")
response = client.chat.completions.create(
    model="auto",  # TokenRouter routes automatically
    messages=[{"role": "user", "content": "Write a sort function"}],
)
print(response.model)  # actual model used
```

### 5. CLI

```bash
# Classify a prompt
tokenrouter classify "Write a Python function to parse JSON"
# Output: {"task_type": "coding", "complexity": "low", "model": "deepseek-chat", ...}

tokenrouter classify "请帮我翻译这段话"
# Output: {"task_type": "chinese_language", ...}
```

## Supported Models

| Provider | Models | Best For |
|----------|--------|----------|
| **OpenAI** | GPT-5.2, GPT-5 Mini | General, coding, reasoning |
| **Anthropic** | Claude Opus 4/4.5, Sonnet 4, Haiku 4.5 | Coding, creative writing, reasoning |
| **Google** | Gemini 3 Flash, 2.5 Flash/Pro | Quick tasks, summarization |
| **DeepSeek** | DeepSeek V3.2, R1 | Coding, math, reasoning (cheapest) |
| **Moonshot** | Kimi K2.5 | Chinese, coding |
| **Qwen** | Turbo, Plus, Max | Chinese language tasks |
| **Zhipu** | GLM-4 Plus | Chinese, general QA |

## How Routing Works

1. **L1 Classifier** — Regex-based heuristic analyzes the prompt for task patterns (coding keywords, math symbols, translation phrases, etc.) and scores 8 task categories
2. **L2 Classifier** — If L1 confidence < 0.7, a cheap model (GPT-5 Mini / Gemini Flash) refines the classification
3. **Routing Table** — Maps `task_type:complexity` to candidate models (24 combinations)
4. **Strategy Selection** — Picks from candidates using the capability matrix:
   - `cheapest`: Lowest cost per token
   - `best`: Highest capability score, then cheapest
   - `balanced`: Best quality/cost ratio
5. **Fallback Chain** — If the primary model fails (429/5xx), automatically retries with the next candidate

## Custom Rules

Override automatic routing for specific task types:

```python
from tokenrouter import TokenRouter
from tokenrouter.types import CustomRule

router = TokenRouter(
    keys={"openai": "sk-...", "deepseek": "sk-..."},
    strategy="balanced",
    rules=[
        CustomRule(task="coding", model="deepseek-chat"),
        CustomRule(task="chinese_language", model="qwen-max"),
    ],
)
```

## Configuration Reference

### `TokenRouter` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keys` | `dict[str, str]` | `{}` | Provider API keys |
| `strategy` | `str` | `"balanced"` | `cheapest`, `best`, or `balanced` |
| `rules` | `list[CustomRule]` | `[]` | Custom routing overrides |
| `exclude_models` | `list[str]` | `[]` | Models to never use |

### Proxy Headers

| Header | Description |
|--------|-------------|
| `X-Routing-Strategy` | Override strategy per request |
| `X-TokenRouter-Model` | Model actually used (response) |
| `X-TokenRouter-Task` | Detected task type (response) |
| `X-TokenRouter-Complexity` | Detected complexity (response) |
| `X-TokenRouter-Cost` | Estimated cost in USD (response) |

## Development

```bash
git clone https://github.com/tokenrouter/tokenrouter.git
cd tokenrouter
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
