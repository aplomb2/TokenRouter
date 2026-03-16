"""Tests for model registry and selection."""

import pytest

from tokenrouter.models import (
    MODEL_REGISTRY,
    ModelCapability,
    ModelConfig,
    calculate_cost,
    get_model,
    select_optimal_model,
    task_type_to_capability,
)


class TestModelRegistry:
    def test_registry_not_empty(self):
        assert len(MODEL_REGISTRY) > 0

    def test_all_models_have_required_fields(self):
        for model in MODEL_REGISTRY:
            assert model.id
            assert model.provider
            assert model.provider_model_id
            assert model.name
            assert model.input_cost_per_1m >= 0
            assert model.output_cost_per_1m >= 0
            assert model.max_tokens > 0
            assert len(model.strengths) > 0
            assert len(model.complexity_range) > 0

    def test_unique_ids(self):
        ids = [m.id for m in MODEL_REGISTRY]
        assert len(ids) == len(set(ids))

    def test_get_model_found(self):
        model = get_model("gpt-5.2")
        assert model is not None
        assert model.provider == "openai"

    def test_get_model_not_found(self):
        assert get_model("nonexistent-model") is None


class TestTaskTypeToCapability:
    def test_complex_reasoning_maps_to_reasoning(self):
        assert task_type_to_capability("complex_reasoning") == "reasoning"

    def test_coding_maps_to_coding(self):
        assert task_type_to_capability("coding") == "coding"

    def test_math_maps_to_math(self):
        assert task_type_to_capability("math") == "math"


class TestCalculateCost:
    def test_zero_tokens(self):
        model = get_model("gpt-5.2")
        assert model is not None
        assert calculate_cost(model, 0, 0) == 0.0

    def test_cost_calculation(self):
        model = get_model("gpt-5.2")
        assert model is not None
        # 1M input + 1M output
        cost = calculate_cost(model, 1_000_000, 1_000_000)
        assert cost == model.input_cost_per_1m + model.output_cost_per_1m

    def test_proportional_cost(self):
        model = get_model("gpt-5-mini")
        assert model is not None
        cost = calculate_cost(model, 500_000, 500_000)
        expected = (model.input_cost_per_1m + model.output_cost_per_1m) / 2
        assert abs(cost - expected) < 0.001


class TestSelectOptimalModel:
    def test_cheapest_strategy(self):
        models = [m for m in MODEL_REGISTRY if m.provider in ("openai", "deepseek")]
        result = select_optimal_model("coding", "medium", models, "cheapest")
        assert result is not None
        # Cheapest should pick a low-cost model
        assert result.input_cost_per_1m + result.output_cost_per_1m <= 5

    def test_best_strategy(self):
        models = [m for m in MODEL_REGISTRY if m.provider in ("openai", "anthropic")]
        result = select_optimal_model("coding", "high", models, "best")
        assert result is not None
        assert result.capabilities.coding >= 4

    def test_balanced_strategy(self):
        result = select_optimal_model("coding", "medium", MODEL_REGISTRY, "balanced")
        assert result is not None

    def test_empty_models(self):
        result = select_optimal_model("coding", "medium", [], "balanced")
        assert result is None

    def test_relaxed_quality(self):
        # Only low-capability models — should still return something
        low_models = [m for m in MODEL_REGISTRY if m.capabilities.coding <= 2]
        result = select_optimal_model("coding", "high", low_models, "balanced")
        assert result is not None

    def test_chinese_language_task(self):
        result = select_optimal_model("chinese_language", "medium", MODEL_REGISTRY, "balanced")
        assert result is not None
        assert result.capabilities.chinese_language >= 3
