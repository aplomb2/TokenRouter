# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TokenRouter is an AI model routing SDK that automatically selects the optimal model based on prompt content. It uses a two-level classification system (L1 regex heuristics, L2 model-based refinement) to categorize tasks and route them to the cheapest appropriate model. Published on PyPI as `byok-router`. Python 3.10+.

## Commands

```bash
# Install for development (includes all optional deps + test tools)
pip install -e ".[dev]"

# Run all tests (182 tests, pytest-asyncio with asyncio_mode="auto")
pytest

# Run a single test file or specific test
pytest tests/test_classifier.py
pytest -k "test_coding"

# Lint and format
ruff check tokenrouter tests
ruff format tokenrouter tests

# Start proxy server
tokenrouter serve --config config.yaml --port 8000

# CLI tools
tokenrouter classify "Write a Python function"
tokenrouter keys create --name "prod" --strategy balanced
tokenrouter keys list
tokenrouter usage 1 --period month
```

## Architecture

**Two deployment modes:** direct SDK usage (`TokenRouter` class) or self-hosted OpenAI-compatible proxy server (FastAPI).

**Request flow:** Messages → Classification (L1→L2) → Routing Table lookup → Strategy selection → Provider adapter call → Fallback on error → Billing → Response with metadata.

### Core Modules (`tokenrouter/`)

- **`__init__.py`** — `TokenRouter` class: main public API with `chat()`, `achat()`, `classify()` methods. Sync methods wrap async via ThreadPoolExecutor.
- **`classifier.py`** — Two-level classification engine. L1 uses regex pattern matching for 8 task types with complexity estimation. L2 uses a cheap model call when L1 confidence < 0.7.
- **`models.py`** — Model registry with 17 models across 8 providers. Each model has capability scores (1-5) on 8 dimensions. Contains the routing table (24 task:complexity combinations) and strategy-based selection (cheapest/best/balanced).
- **`proxy.py`** — FastAPI server providing `/v1/chat/completions` (OpenAI-compatible), key management API, usage/billing endpoints, and dashboard serving.
- **`keys.py`** — SQLite-based unified key management (`tr-xxx` keys). Handles multi-tenant provider key storage with WAL mode and FK constraints.
- **`billing.py`** — Cost tracking per request, baseline comparison (default: claude-opus-4), savings reports.
- **`fallback.py`** — Automatic retry on 429/5xx errors with fallback to alternative models from the routing table.
- **`config.py`** — YAML config loader with `${ENV_VAR}` interpolation.
- **`types.py`** — OpenAI-compatible request/response dataclasses.
- **`providers/`** — Provider adapters (OpenAI, Anthropic, Google, plus OpenAI-compatible generic adapter for DeepSeek/Moonshot/Qwen/Doubao/Zhipu). Each adapter normalizes to a common response format.

### Key Type Definitions

- `TaskType`: `"coding" | "translation" | "simple_qa" | "complex_reasoning" | "creative_writing" | "math" | "summarization" | "chinese_language"`
- `RoutingStrategy`: `"cheapest" | "best" | "balanced"`
- `ComplexityLevel`: `"low" | "medium" | "high"`

## Code Style

- **ruff** for linting and formatting: line-length 120, target Python 3.10
- Async-first design: core logic is async, sync methods are wrappers
- Type hints throughout (PEP 484)
- Tests use `pytest-asyncio` (auto mode) and `respx` for HTTP mocking
