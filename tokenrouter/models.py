"""Model registry, capability matrix, and optimal model selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from tokenrouter.types import (
    CapabilityKey,
    ComplexityLevel,
    ProviderType,
    RoutingStrategy,
    TaskType,
)


@dataclass
class ModelCapability:
    coding: int = 1
    reasoning: int = 1
    translation: int = 1
    simple_qa: int = 1
    creative_writing: int = 1
    math: int = 1
    summarization: int = 1
    chinese_language: int = 1

    def get(self, key: str) -> int:
        return getattr(self, key, 1)


@dataclass
class ModelConfig:
    id: str
    provider: ProviderType
    provider_model_id: str
    name: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    max_tokens: int
    strengths: list[TaskType]
    complexity_range: list[ComplexityLevel]
    capabilities: ModelCapability


def task_type_to_capability(task_type: TaskType) -> CapabilityKey:
    """Map TaskType to CapabilityKey (complex_reasoning -> reasoning)."""
    if task_type == "complex_reasoning":
        return "reasoning"
    return task_type  # type: ignore[return-value]


MODEL_REGISTRY: list[ModelConfig] = [
    # === Anthropic ===
    ModelConfig(
        id="claude-opus-4",
        provider="anthropic",
        provider_model_id="claude-opus-4-20250514",
        name="Claude Opus 4",
        input_cost_per_1m=15,
        output_cost_per_1m=75,
        max_tokens=4096,
        strengths=["complex_reasoning", "coding", "math"],
        complexity_range=["high"],
        capabilities=ModelCapability(coding=5, reasoning=5, translation=4, simple_qa=5, creative_writing=5, math=5, summarization=5, chinese_language=2),
    ),
    ModelConfig(
        id="claude-opus-4.5",
        provider="anthropic",
        provider_model_id="claude-opus-4-5-20251101",
        name="Claude Opus 4.5",
        input_cost_per_1m=5,
        output_cost_per_1m=25,
        max_tokens=8192,
        strengths=["complex_reasoning", "coding", "creative_writing", "math"],
        complexity_range=["medium", "high"],
        capabilities=ModelCapability(coding=5, reasoning=5, translation=4, simple_qa=5, creative_writing=5, math=5, summarization=5, chinese_language=3),
    ),
    ModelConfig(
        id="claude-sonnet-4",
        provider="anthropic",
        provider_model_id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        input_cost_per_1m=3,
        output_cost_per_1m=15,
        max_tokens=4096,
        strengths=["coding", "creative_writing", "complex_reasoning"],
        complexity_range=["medium", "high"],
        capabilities=ModelCapability(coding=4, reasoning=4, translation=5, simple_qa=5, creative_writing=5, math=3, summarization=4, chinese_language=2),
    ),
    ModelConfig(
        id="claude-haiku-4.5",
        provider="anthropic",
        provider_model_id="claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        input_cost_per_1m=1,
        output_cost_per_1m=5,
        max_tokens=4096,
        strengths=["simple_qa", "summarization", "translation"],
        complexity_range=["low", "medium"],
        capabilities=ModelCapability(coding=3, reasoning=3, translation=3, simple_qa=4, creative_writing=3, math=2, summarization=3, chinese_language=1),
    ),
    # === OpenAI ===
    ModelConfig(
        id="gpt-5.2",
        provider="openai",
        provider_model_id="gpt-5.2",
        name="GPT-5.2",
        input_cost_per_1m=1.75,
        output_cost_per_1m=14,
        max_tokens=16384,
        strengths=["coding", "complex_reasoning", "creative_writing", "math"],
        complexity_range=["medium", "high"],
        capabilities=ModelCapability(coding=5, reasoning=5, translation=5, simple_qa=5, creative_writing=5, math=5, summarization=5, chinese_language=3),
    ),
    ModelConfig(
        id="gpt-5-mini",
        provider="openai",
        provider_model_id="gpt-5-mini",
        name="GPT-5 Mini",
        input_cost_per_1m=0.25,
        output_cost_per_1m=2,
        max_tokens=8192,
        strengths=["simple_qa", "translation", "summarization", "coding"],
        complexity_range=["low", "medium"],
        capabilities=ModelCapability(coding=3, reasoning=3, translation=4, simple_qa=4, creative_writing=3, math=3, summarization=4, chinese_language=2),
    ),
    # === Google Gemini ===
    ModelConfig(
        id="gemini-3-flash",
        provider="google",
        provider_model_id="gemini-3-flash-preview",
        name="Gemini 3 Flash",
        input_cost_per_1m=0.50,
        output_cost_per_1m=3.0,
        max_tokens=8192,
        strengths=["simple_qa", "summarization", "coding"],
        complexity_range=["low", "medium"],
        capabilities=ModelCapability(coding=4, reasoning=3, translation=3, simple_qa=4, creative_writing=3, math=3, summarization=4, chinese_language=2),
    ),
    ModelConfig(
        id="gemini-2.5-flash",
        provider="google",
        provider_model_id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        input_cost_per_1m=0.30,
        output_cost_per_1m=2.50,
        max_tokens=8192,
        strengths=["simple_qa", "summarization", "coding"],
        complexity_range=["low", "medium"],
        capabilities=ModelCapability(coding=3, reasoning=3, translation=3, simple_qa=4, creative_writing=3, math=3, summarization=4, chinese_language=1),
    ),
    ModelConfig(
        id="gemini-2.5-pro",
        provider="google",
        provider_model_id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        input_cost_per_1m=1.25,
        output_cost_per_1m=10,
        max_tokens=8192,
        strengths=["coding", "complex_reasoning", "math", "summarization"],
        complexity_range=["medium", "high"],
        capabilities=ModelCapability(coding=4, reasoning=4, translation=3, simple_qa=4, creative_writing=4, math=4, summarization=5, chinese_language=2),
    ),
    # === Chinese AI Models ===
    ModelConfig(
        id="deepseek-chat",
        provider="deepseek",
        provider_model_id="deepseek-chat",
        name="DeepSeek V3.2",
        input_cost_per_1m=0.28,
        output_cost_per_1m=0.42,
        max_tokens=8192,
        strengths=["coding", "complex_reasoning", "math"],
        complexity_range=["medium", "high"],
        capabilities=ModelCapability(coding=4, reasoning=4, translation=3, simple_qa=3, creative_writing=2, math=4, summarization=3, chinese_language=5),
    ),
    ModelConfig(
        id="deepseek-reasoner",
        provider="deepseek",
        provider_model_id="deepseek-reasoner",
        name="DeepSeek R1 (V3.2 Thinking)",
        input_cost_per_1m=0.28,
        output_cost_per_1m=0.42,
        max_tokens=8192,
        strengths=["complex_reasoning", "math", "coding"],
        complexity_range=["high"],
        capabilities=ModelCapability(coding=4, reasoning=5, translation=2, simple_qa=2, creative_writing=1, math=5, summarization=2, chinese_language=5),
    ),
    ModelConfig(
        id="kimi-k2.5",
        provider="moonshot",
        provider_model_id="kimi-k2.5",
        name="Kimi K2.5",
        input_cost_per_1m=0.60,
        output_cost_per_1m=2.50,
        max_tokens=8192,
        strengths=["chinese_language", "coding", "complex_reasoning", "translation"],
        complexity_range=["medium", "high"],
        capabilities=ModelCapability(coding=4, reasoning=4, translation=4, simple_qa=4, creative_writing=3, math=3, summarization=3, chinese_language=5),
    ),
    ModelConfig(
        id="qwen-turbo",
        provider="qwen",
        provider_model_id="qwen-turbo",
        name="Qwen Turbo",
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.20,
        max_tokens=4096,
        strengths=["chinese_language", "simple_qa", "translation"],
        complexity_range=["low"],
        capabilities=ModelCapability(coding=1, reasoning=1, translation=2, simple_qa=2, creative_writing=1, math=1, summarization=2, chinese_language=3),
    ),
    ModelConfig(
        id="qwen-plus",
        provider="qwen",
        provider_model_id="qwen-plus",
        name="Qwen Plus",
        input_cost_per_1m=0.40,
        output_cost_per_1m=1.20,
        max_tokens=4096,
        strengths=["chinese_language", "coding", "complex_reasoning"],
        complexity_range=["medium"],
        capabilities=ModelCapability(coding=2, reasoning=2, translation=3, simple_qa=3, creative_writing=2, math=2, summarization=2, chinese_language=4),
    ),
    ModelConfig(
        id="qwen-max",
        provider="qwen",
        provider_model_id="qwen-max",
        name="Qwen Max",
        input_cost_per_1m=1.60,
        output_cost_per_1m=6.40,
        max_tokens=8192,
        strengths=["chinese_language", "complex_reasoning", "coding", "translation"],
        complexity_range=["medium", "high"],
        capabilities=ModelCapability(coding=2, reasoning=3, translation=4, simple_qa=4, creative_writing=3, math=2, summarization=3, chinese_language=5),
    ),
    ModelConfig(
        id="glm-4-plus",
        provider="zhipu",
        provider_model_id="glm-4-plus",
        name="GLM-4 Plus",
        input_cost_per_1m=0.50,
        output_cost_per_1m=1.50,
        max_tokens=4096,
        strengths=["chinese_language", "simple_qa", "creative_writing"],
        complexity_range=["low", "medium"],
        capabilities=ModelCapability(coding=2, reasoning=2, translation=2, simple_qa=3, creative_writing=2, math=2, summarization=2, chinese_language=4),
    ),
]

_MODEL_INDEX: dict[str, ModelConfig] = {m.id: m for m in MODEL_REGISTRY}


def get_model(model_id: str) -> ModelConfig | None:
    """Look up a model by ID."""
    return _MODEL_INDEX.get(model_id)


def calculate_cost(model: ModelConfig, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a given token count."""
    return (input_tokens / 1_000_000) * model.input_cost_per_1m + (output_tokens / 1_000_000) * model.output_cost_per_1m


def _sort_models(models: list[ModelConfig], cap_key: str, strategy: RoutingStrategy) -> list[ModelConfig]:
    """Sort models by strategy."""
    if strategy == "cheapest":
        return sorted(models, key=lambda m: m.input_cost_per_1m + m.output_cost_per_1m)
    elif strategy == "best":
        return sorted(
            models,
            key=lambda m: (-m.capabilities.get(cap_key), m.input_cost_per_1m + m.output_cost_per_1m),
        )
    else:
        # balanced: quality / cost ratio
        def ratio(m: ModelConfig) -> float:
            cost = m.input_cost_per_1m + m.output_cost_per_1m or 0.01
            return m.capabilities.get(cap_key) / cost
        return sorted(models, key=ratio, reverse=True)


def select_optimal_model(
    task_type: TaskType,
    complexity: ComplexityLevel,
    available_models: list[ModelConfig],
    strategy: RoutingStrategy,
) -> ModelConfig | None:
    """Select the optimal model based on task type, complexity, available models, and strategy."""
    if not available_models:
        return None

    cap_key = task_type_to_capability(task_type)
    thresholds = {"low": 1, "medium": 3, "high": 4}
    min_quality = thresholds.get(complexity, 1)

    qualified = [m for m in available_models if m.capabilities.get(cap_key) >= min_quality]

    if not qualified:
        # Relax quality requirement
        relaxed = [m for m in available_models if m.capabilities.get(cap_key) >= max(1, min_quality - 1)]
        if not relaxed:
            return available_models[0]
        return _sort_models(relaxed, cap_key, strategy)[0]

    return _sort_models(qualified, cap_key, strategy)[0]
