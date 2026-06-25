"""Agent provider configuration.

Stores which LLM provider the agent uses plus its connection settings
(base_url, model, api_key, options). Persisted to a gitignored JSON file under
backend/storage so API keys never reach the browser — the frontend only ever
sees provider/model/base_url metadata via `AgentConfig.public()`.

Resolution order for the API key: stored value → AGENT_API_KEY env →
provider-specific env (OPENAI_API_KEY / ANTHROPIC_API_KEY). That lets users
export a key instead of pasting it into the UI.

The key is stored in plaintext in a local, gitignored file. That is an accepted
tradeoff for a self-hosted, single-user educational tool; do not reuse this for
a multi-tenant deployment.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from paths import STORAGE_DIR

# Lives in the unified repo-root storage/ (anchored in paths.py, independent of the
# process working directory) so API keys never reach the browser.
_CONFIG_PATH = STORAGE_DIR / "agent_config.json"

DEFAULT_PROVIDER = "openai_compat"

# Per-provider env var consulted as a key fallback when none is stored.
_PROVIDER_KEY_ENV = {
    "openai_compat": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@dataclass
class AgentConfig:
    provider: str = DEFAULT_PROVIDER
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    options: dict = field(default_factory=dict)  # e.g. {"temperature": 0.7}

    def resolved_api_key(self) -> str:
        """The key to actually use: stored, then env fallbacks."""
        if self.api_key:
            return self.api_key
        if os.environ.get("AGENT_API_KEY"):
            return os.environ["AGENT_API_KEY"]
        env_name = _PROVIDER_KEY_ENV.get(self.provider)
        if env_name and os.environ.get(env_name):
            return os.environ[env_name]
        return ""

    def public(self) -> dict:
        """Metadata safe to expose to the browser — never the key itself."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "options": self.options,
            "hasApiKey": bool(self.resolved_api_key()),
        }


def load_config() -> AgentConfig:
    """Read the stored config, applying env overrides for provider/model/base_url."""
    data: dict = {}
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}

    for env_key, field_name in (
        ("AGENT_PROVIDER", "provider"),
        ("AGENT_MODEL", "model"),
        ("AGENT_BASE_URL", "base_url"),
    ):
        if os.environ.get(env_key):
            data[field_name] = os.environ[env_key]

    return AgentConfig(
        provider=data.get("provider") or DEFAULT_PROVIDER,
        model=data.get("model", ""),
        base_url=data.get("base_url", ""),
        api_key=data.get("api_key", ""),
        options=data.get("options") or {},
    )


def save_config(partial: dict) -> AgentConfig:
    """Merge a partial settings dict into the stored config and persist it.

    A blank/absent `api_key` leaves the existing key untouched (so the UI can
    re-save other fields without re-entering the key).
    """
    cfg = load_config()
    if partial.get("provider"):
        cfg.provider = partial["provider"]
    if "model" in partial:
        cfg.model = partial.get("model") or ""
    if "base_url" in partial:
        cfg.base_url = partial.get("base_url") or ""
    if partial.get("api_key"):
        cfg.api_key = partial["api_key"]
    if isinstance(partial.get("options"), dict):
        cfg.options = partial["options"]

    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2))
    return cfg
