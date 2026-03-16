"""YAML configuration loader with environment variable interpolation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from tokenrouter.types import CustomRule, RoutingStrategy

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _interpolate_env(value: str) -> str:
    """Replace ${VAR_NAME} with environment variable values."""
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")
    return _ENV_VAR_PATTERN.sub(replacer, value)


@dataclass
class ProxyConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors: bool = True


@dataclass
class TokenRouterConfig:
    keys: dict[str, str] = field(default_factory=dict)
    strategy: RoutingStrategy = "balanced"
    rules: list[CustomRule] = field(default_factory=list)
    exclude_models: list[str] = field(default_factory=list)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    admin_token: str = ""
    database: str = ""
    baseline_model: str = "claude-opus-4"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenRouterConfig:
        """Create config from a dict (e.g. parsed YAML)."""
        # Keys with env var interpolation
        keys = {}
        for k, v in data.get("keys", {}).items():
            keys[k] = _interpolate_env(str(v)) if isinstance(v, str) else str(v)

        strategy = data.get("strategy", "balanced")

        rules = []
        for r in data.get("rules", []):
            rules.append(CustomRule(task=r.get("task", "*"), model=r.get("model", "")))

        exclude = data.get("exclude_models", [])

        proxy_data = data.get("proxy", {})
        proxy = ProxyConfig(
            host=proxy_data.get("host", "0.0.0.0"),
            port=proxy_data.get("port", 8000),
            cors=proxy_data.get("cors", True),
        )

        # New fields
        admin_token_raw = data.get("admin_token", "")
        admin_token = _interpolate_env(str(admin_token_raw)) if admin_token_raw else ""

        database = data.get("database", "")
        if database:
            database = os.path.expanduser(str(database))

        baseline_model = data.get("baseline_model", "claude-opus-4")

        return cls(
            keys=keys,
            strategy=strategy,
            rules=rules,
            exclude_models=exclude,
            proxy=proxy,
            admin_token=admin_token,
            database=database,
            baseline_model=baseline_model,
        )

    @classmethod
    def from_yaml(cls, path: str) -> TokenRouterConfig:
        """Load config from a YAML file. Requires pyyaml."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config. Install with: pip install tokenrouter[yaml]"
            )
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})
