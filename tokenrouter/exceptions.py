"""Custom exception types for TokenRouter."""


class TokenRouterError(Exception):
    """Base exception for all TokenRouter errors."""


class ProviderError(TokenRouterError):
    """Provider API call failed (e.g. HTTP error from OpenAI, Anthropic, etc.)."""

    def __init__(self, provider: str, status_code: int, detail: str) -> None:
        self.provider = provider
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{provider} API error {status_code}: {detail}")


class ClassificationError(TokenRouterError):
    """Task classification failed."""


class ConfigError(TokenRouterError):
    """Configuration error (invalid YAML, missing fields, etc.)."""


class AuthError(TokenRouterError):
    """Authentication or API key error."""
