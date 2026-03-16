"""L1 (heuristic) and L2 (model-based) task classifier with routing table."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from tokenrouter.models import (
    MODEL_REGISTRY,
    ModelConfig,
    get_model,
    select_optimal_model,
    task_type_to_capability,
)
from tokenrouter.types import (
    ComplexityLevel,
    CustomRule,
    RoutingStrategy,
    TaskType,
    extract_text,
)

# ========== Pattern definitions ==========

CODING_PATTERNS = re.compile(
    r"\b(code|function|implement|debug|refactor|algorithm|class|method|api|endpoint|component"
    r"|typescript|python|javascript|rust|golang|sql|html|css|react|vue|angular|regex|compile"
    r"|runtime|error|bug|fix|deploy|docker|kubernetes|git|npm|pip|cargo|async|await|promise"
    r"|callback|interface|struct|enum|loop|array|hashmap|linked.?list|binary.?tree|sort|merge"
    r"|stack|queue|pointer|memory|heap|thread|mutex|database|query|migration|schema|ORM|REST"
    r"|GraphQL|websocket|middleware|authentication|authorization|JWT|OAuth|CRUD|MVC|singleton"
    r"|factory|observer|decorator|proxy|adapter)\b",
    re.IGNORECASE,
)

MATH_PATTERNS = re.compile(
    r"\b(calculate|equation|formula|integral|derivative|matrix|probability|statistics|algebra"
    r"|geometry|trigonometry|calculus|proof|theorem|solve|compute|multiply|divide|factorial"
    r"|logarithm|polynomial|eigenvalue|determinant|vector|gradient|optimization|regression"
    r"|bayesian|hypothesis|variance|standard.?deviation|mean|median|mode|permutation|combination"
    r"|binomial|normal.?distribution|chi.?square|p-value)\b",
    re.IGNORECASE,
)

TRANSLATION_PATTERNS = re.compile(
    r"\b(translate|翻译|traduire|übersetzen|traducir|переведи|번역"
    r"|into\s+(english|chinese|spanish|french|german|japanese|korean|arabic|portuguese"
    r"|russian|italian|hindi|thai|vietnamese|indonesian|turkish|dutch|polish|swedish))\b",
    re.IGNORECASE,
)

CREATIVE_PATTERNS = re.compile(
    r"\b(write a story|poem|creative|fiction|novel|narrative|screenplay|dialogue|character|plot"
    r"|imagine|fantasy|compose|lyrics|haiku|sonnet|write me|blog.?post|article|essay|copywriting"
    r"|slogan|tagline|brand.?voice|tone|rewrite.?this|make.?it.?sound|storytelling)\b",
    re.IGNORECASE,
)

SUMMARIZATION_PATTERNS = re.compile(
    r"\b(summarize|summary|summarise|tldr|tl;dr|brief|key points|main points|condense|shorten"
    r"|overview|recap|bullet.?points|extract.?key|boil.?down|digest)\b",
    re.IGNORECASE,
)

REASONING_PATTERNS = re.compile(
    r"\b(analyze|compare|evaluate|pros and cons|trade-?offs|why|reason|argue|debate|implications"
    r"|consequences|philosophy|ethics|explain why|explain how|in depth|detailed analysis|critique"
    r"|assess|weigh|consider|think.?through|step.?by.?step|logical|inference|deduce|conclude"
    r"|evidence|argument)\b",
    re.IGNORECASE,
)

CODE_LANG_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(
        r"\b(python|pip|django|flask|pandas|numpy|pytorch|tensorflow|def\s+\w+|import\s+\w+"
        r"|\.py\b|f-string|list comprehension|__init__|virtualenv|conda)\b", re.IGNORECASE,
    ),
    "javascript": re.compile(
        r"\b(javascript|js|node\.?js|npm|yarn|react|vue|angular|express|next\.?js|webpack|babel"
        r"|eslint|const\s+\w+|let\s+\w+|=>|\.jsx?\b|async\/await|promise\.all)\b", re.IGNORECASE,
    ),
    "typescript": re.compile(
        r"\b(typescript|ts|\.tsx?\b|interface\s+\w+|type\s+\w+|generic|keyof|typeof|as\s+const"
        r"|enum\s+\w+)\b", re.IGNORECASE,
    ),
    "sql": re.compile(
        r"\b(sql|select\s+\w|insert\s+into|update\s+\w+\s+set|delete\s+from|join\s+\w+"
        r"|where\s+\w+|group\s+by|order\s+by|having|index|foreign\s+key|primary\s+key"
        r"|alter\s+table|create\s+table)\b", re.IGNORECASE,
    ),
    "rust": re.compile(
        r"\b(rust|cargo|fn\s+\w+|impl\s+\w+|trait\s+\w+|let\s+mut|Option<|Result<|unwrap"
        r"|borrow|lifetime|\.rs\b|tokio|async-std)\b", re.IGNORECASE,
    ),
    "go": re.compile(
        r"\b(golang|go\s+mod|func\s+\w+|goroutine|channel|defer|panic|recover|\.go\b"
        r"|go\s+run|go\s+build)\b", re.IGNORECASE,
    ),
    "java": re.compile(
        r"\b(java\b|public\s+class|private\s+\w+|protected\s+\w+|@Override|@Autowired|Spring"
        r"|Maven|Gradle|\.java\b|JVM|garbage.?collection)\b", re.IGNORECASE,
    ),
    "cpp": re.compile(
        r"\b(c\+\+|cpp|std::|iostream|vector<|template<|namespace|#include|\.cpp\b|\.hpp\b"
        r"|malloc|free|pointer|virtual)\b", re.IGNORECASE,
    ),
    "swift": re.compile(
        r"\b(swift|SwiftUI|UIKit|Xcode|\.swift\b|var\s+\w+:\s*\w+|let\s+\w+:\s*\w+"
        r"|@State|@Binding|protocol\s+\w+|struct\s+\w+:\s*View)\b", re.IGNORECASE,
    ),
}

LANGUAGE_PATTERNS: dict[str, re.Pattern[str]] = {
    "chinese": re.compile(r"[\u4e00-\u9fff]{2,}"),
    "japanese": re.compile(r"[\u3040-\u309f\u30a0-\u30ff]{2,}"),
    "korean": re.compile(r"[\uac00-\ud7af]{2,}"),
    "arabic": re.compile(r"[\u0600-\u06ff]{3,}"),
    "russian": re.compile(r"[\u0400-\u04ff]{3,}"),
    "thai": re.compile(r"[\u0e00-\u0e7f]{3,}"),
}

# ========== Routing table ==========

ROUTING_TABLE: dict[str, list[str]] = {
    "simple_qa:low": ["qwen-turbo", "gemini-2.5-flash", "gpt-5-mini", "claude-haiku-4.5"],
    "simple_qa:medium": ["gpt-5-mini", "claude-haiku-4.5", "gemini-3-flash", "gpt-5.2"],
    "simple_qa:high": ["gpt-5.2", "claude-sonnet-4", "gemini-2.5-pro"],
    "translation:low": ["qwen-turbo", "gemini-2.5-flash", "gpt-5-mini", "claude-haiku-4.5"],
    "translation:medium": ["kimi-k2.5", "gpt-5-mini", "gpt-5.2", "claude-haiku-4.5"],
    "translation:high": ["gpt-5.2", "qwen-max", "claude-sonnet-4"],
    "coding:low": ["deepseek-chat", "gpt-5-mini", "gemini-3-flash", "claude-haiku-4.5"],
    "coding:medium": ["deepseek-chat", "claude-sonnet-4", "gpt-5.2", "gemini-2.5-pro"],
    "coding:high": ["gpt-5.2", "claude-opus-4.5", "deepseek-chat", "claude-sonnet-4"],
    "creative_writing:low": ["gpt-5-mini", "claude-haiku-4.5", "gemini-2.5-flash"],
    "creative_writing:medium": ["claude-sonnet-4", "gpt-5.2", "gemini-2.5-pro"],
    "creative_writing:high": ["claude-sonnet-4", "gpt-5.2", "claude-opus-4.5"],
    "complex_reasoning:low": ["deepseek-chat", "gpt-5.2", "claude-sonnet-4", "gemini-2.5-pro"],
    "complex_reasoning:medium": ["deepseek-chat", "claude-sonnet-4", "gpt-5.2", "gemini-2.5-pro"],
    "complex_reasoning:high": ["deepseek-reasoner", "claude-opus-4.5", "claude-opus-4", "gpt-5.2"],
    "math:low": ["deepseek-chat", "gpt-5.2", "gemini-2.5-pro", "claude-sonnet-4"],
    "math:medium": ["deepseek-reasoner", "gpt-5.2", "gemini-2.5-pro", "claude-opus-4.5"],
    "math:high": ["deepseek-reasoner", "claude-opus-4", "gpt-5.2", "claude-opus-4.5"],
    "summarization:low": ["qwen-turbo", "gemini-2.5-flash", "claude-haiku-4.5", "gpt-5-mini"],
    "summarization:medium": ["claude-haiku-4.5", "gemini-3-flash", "gpt-5-mini", "gpt-5.2"],
    "summarization:high": ["gpt-5.2", "claude-sonnet-4", "gemini-2.5-pro"],
    "chinese_language:low": ["qwen-turbo", "glm-4-plus", "gemini-2.5-flash"],
    "chinese_language:medium": ["qwen-plus", "kimi-k2.5", "glm-4-plus"],
    "chinese_language:high": ["qwen-max", "kimi-k2.5", "deepseek-chat"],
}

DEFAULT_CANDIDATES = ["gpt-5.2", "claude-sonnet-4", "gpt-5-mini"]

# ========== L2 Classifier ==========

L2_PROMPT = (
    'Classify this task. Reply with JSON only: '
    '{"taskType":"coding|translation|simple_qa|complex_reasoning|creative_writing|math|summarization|chinese_language",'
    '"complexity":"low|medium|high"}\n\nTask: '
)

VALID_TASK_TYPES = {
    "coding", "translation", "simple_qa", "complex_reasoning",
    "creative_writing", "math", "summarization", "chinese_language",
}
VALID_COMPLEXITIES = {"low", "medium", "high"}


def _parse_l2_response(text: str | None) -> tuple[TaskType, ComplexityLevel] | None:
    if not text:
        return None
    match = re.search(r"\{[^}]+\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        if parsed.get("taskType") in VALID_TASK_TYPES and parsed.get("complexity") in VALID_COMPLEXITIES:
            return parsed["taskType"], parsed["complexity"]
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# ========== Detection helpers ==========

def _detect_language(text: str) -> str | None:
    for lang, pattern in LANGUAGE_PATTERNS.items():
        if pattern.search(text):
            return lang
    return None


def _detect_code_language(text: str) -> str | None:
    for lang, pattern in CODE_LANG_PATTERNS.items():
        if pattern.search(text):
            return lang
    return None


# ========== L1 Classifier ==========

@dataclass
class ClassificationResult:
    task_type: TaskType
    complexity: ComplexityLevel
    selected_model: ModelConfig
    reasoning: str
    confidence: float
    fallback_chain: list[str]
    classifier_used: str = "L1"
    user_language: str | None = None
    code_language: str | None = None


def _classify_task_l1(
    messages: list[dict[str, Any]],
) -> tuple[TaskType, ComplexityLevel, str, float, str | None, str | None]:
    """L1 heuristic classifier. Returns (task_type, complexity, reasoning, confidence, code_lang, user_lang)."""
    last_user_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg
            break

    if not last_user_msg:
        return "simple_qa", "low", "No user message found", 0.5, None, None

    content = extract_text(last_user_msg.get("content", ""))
    word_count = len(content.split())
    line_count = content.count("\n") + 1
    total_messages = len(messages)

    # Recent context
    recent_context = " ".join(extract_text(m.get("content", "")) for m in messages[-6:])

    user_language = _detect_language(content)
    code_language = _detect_code_language(content) or _detect_code_language(recent_context)

    # Score each category
    scores: dict[str, float] = {
        "coding": 0,
        "math": 0,
        "translation": 0,
        "creative_writing": 0,
        "summarization": 0,
        "complex_reasoning": 0,
        "simple_qa": 0.3,
        "chinese_language": 0,
    }

    # Chinese language content (not translation)
    if user_language == "chinese" and not TRANSLATION_PATTERNS.search(content):
        scores["chinese_language"] += 0.7

    if CODING_PATTERNS.search(content):
        scores["coding"] += 0.6
    if CODING_PATTERNS.search(recent_context):
        scores["coding"] += 0.2
    if code_language:
        scores["coding"] += 0.3
    if re.search(r"```", content):
        scores["coding"] += 0.4

    if MATH_PATTERNS.search(content):
        scores["math"] += 0.7
    if re.search(r"[=+\-*/^]{2,}|\\frac|\\sum|\\int", content):
        scores["math"] += 0.3

    if TRANSLATION_PATTERNS.search(content):
        scores["translation"] += 0.8
    if user_language and len(content) < 500:
        scores["translation"] += 0.2

    if CREATIVE_PATTERNS.search(content):
        scores["creative_writing"] += 0.7
    if SUMMARIZATION_PATTERNS.search(content):
        scores["summarization"] += 0.8
    if REASONING_PATTERNS.search(content):
        scores["complex_reasoning"] += 0.6

    # Multi-turn boost
    if total_messages > 2:
        prev_msgs = " ".join(extract_text(m.get("content", "")) for m in messages[:-1])
        if CODING_PATTERNS.search(prev_msgs):
            scores["coding"] += 0.15
        if MATH_PATTERNS.search(prev_msgs):
            scores["math"] += 0.1
        if REASONING_PATTERNS.search(prev_msgs):
            scores["complex_reasoning"] += 0.1

    # Find best match
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score = sorted_scores[0]
    _, second_score = sorted_scores[1]

    confidence = min(1.0, best_score)
    task_type: TaskType = best_type  # type: ignore[assignment]

    reasoning = f"Classified as {task_type} (score: {best_score:.2f})"
    if code_language:
        reasoning += f", code lang: {code_language}"
    if user_language:
        reasoning += f", user lang: {user_language}"

    # Adjust confidence if top two are close
    if best_score - second_score < 0.15:
        confidence *= 0.6

    # Complexity
    complexity: ComplexityLevel = "low"
    if word_count > 500 or line_count > 30 or total_messages > 10:
        complexity = "high"
    elif word_count > 100 or line_count > 10 or total_messages > 4:
        complexity = "medium"

    if task_type == "complex_reasoning" and complexity == "low":
        complexity = "medium"
    if task_type == "coding" and code_language and complexity == "low" and word_count > 50:
        complexity = "medium"

    reasoning += f"; {complexity} complexity"
    return task_type, complexity, reasoning, confidence, code_language, user_language


# ========== L2 Classifier (async, uses cheap model) ==========

async def _classify_with_l2(
    user_message: str,
    keys: dict[str, str],
) -> tuple[TaskType, ComplexityLevel] | None:
    """Use a cheap model for more accurate classification when L1 confidence is low."""
    truncated = user_message[:500]
    prompt = L2_PROMPT + truncated

    # Priority: openai → google → deepseek
    provider_configs = [
        ("openai", "gpt-5-mini", "https://api.openai.com/v1"),
        ("google", "gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta/openai"),
        ("deepseek", "deepseek-chat", "https://api.deepseek.com/v1"),
        ("anthropic", "claude-haiku-4-5-20251001", None),  # special handling
    ]

    for provider, model, base_url in provider_configs:
        api_key = keys.get(provider)
        if not api_key:
            continue

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                if provider == "anthropic":
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "max_tokens": 80,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    text = data.get("content", [{}])[0].get("text")
                else:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "max_tokens": 80,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content")

                result = _parse_l2_response(text)
                if result:
                    return result
        except (httpx.HTTPError, KeyError, IndexError):
            continue

    return None


# ========== Main classification + routing ==========

def _select_model_by_strategy(
    candidates: list[str],
    strategy: RoutingStrategy,
    task_type: TaskType,
    complexity: ComplexityLevel,
    custom_rules: list[CustomRule] | None = None,
) -> str:
    """Select a model from candidates using the given strategy."""
    # Check custom rules first
    if custom_rules:
        for rule in custom_rules:
            if rule.task == task_type or rule.task == "*":
                model = get_model(rule.model)
                if model:
                    return model.id

    # Use capability matrix
    candidate_models = [m for mid in candidates if (m := get_model(mid)) is not None]
    if not candidate_models:
        return candidates[0] if candidates else "gpt-5.2"

    optimal = select_optimal_model(task_type, complexity, candidate_models, strategy)
    return optimal.id if optimal else candidates[0]


def classify_sync(
    messages: list[dict[str, Any]],
    strategy: RoutingStrategy = "balanced",
    custom_rules: list[CustomRule] | None = None,
    exclude_models: list[str] | None = None,
) -> ClassificationResult:
    """Synchronous L1-only classification and routing."""
    task_type, complexity, reasoning, confidence, code_lang, user_lang = _classify_task_l1(messages)

    key = f"{task_type}:{complexity}"
    candidates = ROUTING_TABLE.get(key, DEFAULT_CANDIDATES)

    if exclude_models:
        excluded = set(exclude_models)
        candidates = [c for c in candidates if c not in excluded]
        if not candidates:
            candidates = DEFAULT_CANDIDATES

    effective = candidates if confidence >= 0.35 else DEFAULT_CANDIDATES

    selected_id = _select_model_by_strategy(effective, strategy, task_type, complexity, custom_rules)
    selected_model = get_model(selected_id) or MODEL_REGISTRY[0]

    fallback_chain = [c for c in effective if c != selected_id]

    return ClassificationResult(
        task_type=task_type,
        complexity=complexity,
        selected_model=selected_model,
        reasoning=f"{reasoning} -> {strategy} strategy -> {selected_model.name}",
        confidence=confidence,
        fallback_chain=fallback_chain,
        classifier_used="L1",
        user_language=user_lang,
        code_language=code_lang,
    )


async def classify_async(
    messages: list[dict[str, Any]],
    strategy: RoutingStrategy = "balanced",
    custom_rules: list[CustomRule] | None = None,
    keys: dict[str, str] | None = None,
    exclude_models: list[str] | None = None,
) -> ClassificationResult:
    """Async classification with L1 + L2 fallback."""
    task_type, complexity, reasoning, confidence, code_lang, user_lang = _classify_task_l1(messages)
    classifier_used = "L1"

    # L2: if L1 confidence < 0.7, try a cheap model
    if confidence < 0.7 and keys:
        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg
                break
        if last_user_msg:
            l2_result = await _classify_with_l2(
                extract_text(last_user_msg.get("content", "")),
                keys,
            )
            if l2_result:
                l2_task, l2_complexity = l2_result
                reasoning = (
                    f"L2 override: {l2_task}/{l2_complexity} "
                    f"(L1 was {task_type}/{complexity} conf={confidence:.2f})"
                )
                task_type = l2_task
                complexity = l2_complexity
                confidence = 0.85
                classifier_used = "L2"

    key = f"{task_type}:{complexity}"
    candidates = ROUTING_TABLE.get(key, DEFAULT_CANDIDATES)

    if exclude_models:
        excluded = set(exclude_models)
        candidates = [c for c in candidates if c not in excluded]
        if not candidates:
            candidates = DEFAULT_CANDIDATES

    effective = candidates if confidence >= 0.35 else DEFAULT_CANDIDATES

    selected_id = _select_model_by_strategy(effective, strategy, task_type, complexity, custom_rules)
    selected_model = get_model(selected_id) or MODEL_REGISTRY[0]

    fallback_chain = [c for c in effective if c != selected_id]

    return ClassificationResult(
        task_type=task_type,
        complexity=complexity,
        selected_model=selected_model,
        reasoning=f"{reasoning} -> {strategy} strategy -> {selected_model.name}",
        confidence=confidence,
        fallback_chain=fallback_chain,
        classifier_used=classifier_used,
        user_language=user_lang,
        code_language=code_lang,
    )
