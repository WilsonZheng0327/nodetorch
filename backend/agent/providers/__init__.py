"""Provider package — importing it registers all built-in adapters.

Add a provider: create an adapter module that calls `register(...)`, then import
it here so the registration runs.
"""
from agent.providers import anthropic, openai_compat  # noqa: F401  (register side-effects)
from agent.providers.registry import create, list_providers, register

__all__ = ["create", "list_providers", "register"]
