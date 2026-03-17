"""Provider adapters for various LLM APIs."""

from __future__ import annotations

from tokenrouter.types import ProviderAdapter, ProviderType

PROVIDER_KEY_MAP: dict[str, list[str]] = {
    "openai": ["openai"],
    "anthropic": ["anthropic"],
    "google": ["google"],
    "deepseek": ["deepseek"],
    "moonshot": ["moonshot"],
    "qwen": ["qwen", "dashscope"],
    "doubao": ["doubao"],
    "zhipu": ["zhipu"],
}

CHINESE_PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "name": "DeepSeek",
    },
    "moonshot": {
        "env_key": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.ai/v1",
        "name": "Moonshot",
    },
    "qwen": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "name": "Qwen",
    },
    "doubao": {
        "env_key": "DOUBAO_API_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "name": "Doubao",
    },
    "zhipu": {
        "env_key": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "name": "GLM",
    },
}


def create_provider(provider: ProviderType, api_key: str) -> ProviderAdapter:
    """Create a provider adapter for the given provider type and API key."""
    from tokenrouter.providers.anthropic import AnthropicAdapter
    from tokenrouter.providers.google import GoogleAdapter
    from tokenrouter.providers.openai import OpenAIAdapter
    from tokenrouter.providers.openai_compatible import OpenAICompatibleAdapter

    if provider == "openai":
        return OpenAIAdapter(api_key)
    elif provider == "anthropic":
        return AnthropicAdapter(api_key)
    elif provider == "google":
        return GoogleAdapter(api_key)
    elif provider in CHINESE_PROVIDERS:
        config = CHINESE_PROVIDERS[provider]
        return OpenAICompatibleAdapter(api_key, config["base_url"], config["name"])
    else:
        raise ValueError(f"Unknown provider: {provider}")
