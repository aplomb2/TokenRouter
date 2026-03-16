"""Type definitions for TokenRouter — OpenAI-compatible request/response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol

RoutingStrategy = Literal["cheapest", "best", "balanced"]
TaskType = Literal[
    "coding", "translation", "simple_qa", "complex_reasoning",
    "creative_writing", "math", "summarization", "chinese_language",
]
ComplexityLevel = Literal["low", "medium", "high"]
ProviderType = Literal[
    "openai", "anthropic", "google",
    "deepseek", "moonshot", "qwen", "doubao", "zhipu",
]
CapabilityKey = Literal[
    "coding", "reasoning", "translation", "simple_qa",
    "creative_writing", "math", "summarization", "chinese_language",
]

# --- Messages ---

MessageContent = str | list[dict[str, Any]]


def extract_text(content: MessageContent) -> str:
    """Extract plain text from message content (string or content parts array)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content) if content else ""


@dataclass
class ChatMessage:
    role: str
    content: MessageContent


@dataclass
class ChatCompletionRequest:
    messages: list[dict[str, Any]]
    model: str = "auto"
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    strategy: RoutingStrategy | None = None


@dataclass
class ChatCompletionChoice:
    index: int
    message: dict[str, Any]
    finish_reason: str


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class TokenRouterMetadata:
    request_id: str = ""
    model_requested: str = ""
    model_used: str = ""
    task_type: str = ""
    complexity: str = ""
    confidence: float = 0.0
    cost: float = 0.0
    cost_if_opus: float = 0.0
    latency_ms: int = 0
    strategy: str = ""
    fallback: bool = False


@dataclass
class ChatCompletionResponse:
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatCompletionChoice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    _tokenrouter: TokenRouterMetadata | None = None

    @property
    def model_used(self) -> str:
        return self._tokenrouter.model_used if self._tokenrouter else self.model

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {"index": c.index, "message": c.message, "finish_reason": c.finish_reason}
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }
        if self._tokenrouter:
            d["_tokenrouter"] = {
                "request_id": self._tokenrouter.request_id,
                "model_requested": self._tokenrouter.model_requested,
                "model_used": self._tokenrouter.model_used,
                "task_type": self._tokenrouter.task_type,
                "complexity": self._tokenrouter.complexity,
                "confidence": self._tokenrouter.confidence,
                "cost": self._tokenrouter.cost,
                "cost_if_opus": self._tokenrouter.cost_if_opus,
                "latency_ms": self._tokenrouter.latency_ms,
                "strategy": self._tokenrouter.strategy,
                "fallback": self._tokenrouter.fallback,
            }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatCompletionResponse:
        choices = [
            ChatCompletionChoice(
                index=c.get("index", 0),
                message=c.get("message", {}),
                finish_reason=c.get("finish_reason", "stop"),
            )
            for c in data.get("choices", [])
        ]
        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
        )


@dataclass
class ChatCompletionChunk:
    id: str
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": self.choices,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatCompletionChunk:
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion.chunk"),
            created=data.get("created", 0),
            model=data.get("model", ""),
            choices=data.get("choices", []),
        )


@dataclass
class CustomRule:
    task: str  # TaskType or "*"
    model: str  # model id


class ProviderAdapter(Protocol):
    async def chat(
        self, request: ChatCompletionRequest, model_id: str
    ) -> ChatCompletionResponse: ...

    async def chat_stream(
        self, request: ChatCompletionRequest, model_id: str
    ) -> AsyncIterator[ChatCompletionChunk]: ...
