"""Tests for the L1 task classifier."""

from tokenrouter.classifier import classify_sync, _classify_task_l1, _detect_language, _detect_code_language


class TestLanguageDetection:
    def test_detect_chinese(self):
        assert _detect_language("请帮我写一个排序算法") == "chinese"

    def test_detect_japanese(self):
        assert _detect_language("おはようございます") == "japanese"

    def test_detect_korean(self):
        assert _detect_language("안녕하세요 세계") == "korean"

    def test_detect_english(self):
        assert _detect_language("Hello world") is None

    def test_detect_russian(self):
        assert _detect_language("Привет мир как дела") == "russian"


class TestCodeLanguageDetection:
    def test_detect_python(self):
        assert _detect_code_language("Write a Python function") == "python"

    def test_detect_rust(self):
        assert _detect_code_language("implement a trait in Rust") == "rust"

    def test_detect_javascript(self):
        assert _detect_code_language("create a React component") == "javascript"

    def test_detect_typescript(self):
        assert _detect_code_language("define an interface in TypeScript") == "typescript"

    def test_detect_none(self):
        assert _detect_code_language("What is the weather today?") is None


class TestL1Classifier:
    def test_coding_task(self):
        messages = [{"role": "user", "content": "Write a Python function to sort a list using quicksort"}]
        task_type, complexity, *_ = _classify_task_l1(messages)
        assert task_type == "coding"

    def test_math_task(self):
        messages = [{"role": "user", "content": "Calculate the integral of x^2 from 0 to 1"}]
        task_type, *_ = _classify_task_l1(messages)
        assert task_type == "math"

    def test_translation_task(self):
        messages = [{"role": "user", "content": "Translate this into French: Hello, how are you?"}]
        task_type, *_ = _classify_task_l1(messages)
        assert task_type == "translation"

    def test_creative_writing_task(self):
        messages = [{"role": "user", "content": "Write a story about a lonely lighthouse keeper"}]
        task_type, *_ = _classify_task_l1(messages)
        assert task_type == "creative_writing"

    def test_summarization_task(self):
        messages = [{"role": "user", "content": "Summarize the key points of this article"}]
        task_type, *_ = _classify_task_l1(messages)
        assert task_type == "summarization"

    def test_reasoning_task(self):
        messages = [
            {
                "role": "user",
                "content": "Analyze the pros and cons of microservices vs monolith architecture and provide a detailed analysis",
            }
        ]
        task_type, *_ = _classify_task_l1(messages)
        assert task_type == "complex_reasoning"

    def test_simple_qa(self):
        messages = [{"role": "user", "content": "What color is the sky?"}]
        task_type, *_ = _classify_task_l1(messages)
        assert task_type == "simple_qa"

    def test_chinese_language(self):
        messages = [{"role": "user", "content": "请帮我写一篇关于人工智能的文章"}]
        task_type, *_ = _classify_task_l1(messages)
        assert task_type == "chinese_language"

    def test_no_user_message(self):
        messages = [{"role": "system", "content": "You are helpful"}]
        task_type, complexity, *_ = _classify_task_l1(messages)
        assert task_type == "simple_qa"
        assert complexity == "low"

    def test_complexity_high(self):
        long_content = "Write a Python function " + "word " * 600
        messages = [{"role": "user", "content": long_content}]
        _, complexity, *_ = _classify_task_l1(messages)
        assert complexity == "high"

    def test_complexity_medium(self):
        medium_content = "Write a function " + "word " * 150
        messages = [{"role": "user", "content": medium_content}]
        _, complexity, *_ = _classify_task_l1(messages)
        assert complexity == "medium"

    def test_multi_turn_boost(self):
        messages = [
            {"role": "user", "content": "I'm working on a Python project with Django"},
            {"role": "assistant", "content": "Sure, I can help with that."},
            {"role": "user", "content": "Now help me with the next part"},
        ]
        task_type, *_ = _classify_task_l1(messages)
        assert task_type == "coding"


class TestClassifySync:
    def test_basic_routing(self):
        messages = [{"role": "user", "content": "Write a Python function to sort a list"}]
        result = classify_sync(messages)
        assert result.task_type == "coding"
        assert result.selected_model is not None
        assert result.confidence > 0
        assert result.classifier_used == "L1"

    def test_strategy_cheapest(self):
        messages = [{"role": "user", "content": "Write a Python function"}]
        result = classify_sync(messages, strategy="cheapest")
        assert result.selected_model is not None

    def test_strategy_best(self):
        messages = [{"role": "user", "content": "Write a Python function"}]
        result = classify_sync(messages, strategy="best")
        assert result.selected_model is not None

    def test_exclude_models(self):
        messages = [{"role": "user", "content": "What is 2+2?"}]
        result = classify_sync(messages, exclude_models=["qwen-turbo", "gemini-2.5-flash"])
        assert result.selected_model.id not in ["qwen-turbo", "gemini-2.5-flash"]

    def test_custom_rules(self):
        from tokenrouter.types import CustomRule

        messages = [{"role": "user", "content": "Write a Python function"}]
        rules = [CustomRule(task="coding", model="deepseek-chat")]
        result = classify_sync(messages, custom_rules=rules)
        assert result.selected_model.id == "deepseek-chat"

    def test_fallback_chain_populated(self):
        messages = [{"role": "user", "content": "Write a function"}]
        result = classify_sync(messages)
        assert isinstance(result.fallback_chain, list)
        assert result.selected_model.id not in result.fallback_chain
