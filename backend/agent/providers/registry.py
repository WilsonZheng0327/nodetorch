"""Pluggable provider registry.

Adding a provider = write an adapter and `register()` it. `fields` describes the
settings form the frontend renders (which inputs the provider needs).
"""
from __future__ import annotations

from typing import Callable

from agent.config import AgentConfig
from agent.providers.base import LLMProvider, ProviderError

# name -> {factory, label, fields}
_REGISTRY: dict[str, dict] = {}


def register(
    name: str,
    factory: Callable[[AgentConfig], LLMProvider],
    *,
    label: str,
    fields: list[dict],
) -> None:
    _REGISTRY[name] = {"factory": factory, "label": label, "fields": fields}


def create(config: AgentConfig) -> LLMProvider:
    entry = _REGISTRY.get(config.provider)
    if not entry:
        available = ", ".join(_REGISTRY) or "none"
        raise ProviderError(
            f"Unknown provider '{config.provider}'. Available: {available}"
        )
    return entry["factory"](config)


def list_providers() -> list[dict]:
    """Provider metadata for the settings UI (name, label, config fields)."""
    return [
        {"name": name, "label": entry["label"], "fields": entry["fields"]}
        for name, entry in _REGISTRY.items()
    ]
