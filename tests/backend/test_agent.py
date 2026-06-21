"""Tests for the AI agent backend (backend/agent/).

Uses asyncio.run() directly so no pytest-asyncio dependency is needed, and a
stub provider so nothing hits a real LLM.
"""
import asyncio

import pytest

from agent import agent as agent_mod
from agent.catalog import format_catalog
from agent.config import AgentConfig, load_config, save_config
from agent.context import summarize_graph
from agent.providers import create, list_providers
from agent.providers.base import LLMProvider, ProviderError, TextDelta
from agent.providers.registry import register
from agent.session import get_session, reset_session


def test_config_roundtrip_hides_key(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.config._CONFIG_PATH", tmp_path / "agent_config.json")
    for k in (
        "AGENT_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "AGENT_PROVIDER", "AGENT_MODEL", "AGENT_BASE_URL",
    ):
        monkeypatch.delenv(k, raising=False)

    cfg = save_config(
        {"provider": "openai_compat", "model": "gpt-x", "base_url": "http://h/v1", "api_key": "secret"}
    )
    pub = cfg.public()
    assert pub["hasApiKey"] is True
    assert "api_key" not in pub and "secret" not in str(pub)

    # Blank key on re-save keeps the existing key; other fields update.
    cfg2 = save_config({"provider": "openai_compat", "model": "gpt-y"})
    assert cfg2.resolved_api_key() == "secret"
    assert cfg2.model == "gpt-y"

    loaded = load_config()
    assert loaded.model == "gpt-y" and loaded.resolved_api_key() == "secret"


def test_registry_lists_and_creates():
    names = [p["name"] for p in list_providers()]
    assert "openai_compat" in names and "anthropic" in names
    assert isinstance(create(AgentConfig(provider="openai_compat", model="x")), LLMProvider)
    with pytest.raises(ProviderError):
        create(AgentConfig(provider="does-not-exist"))


def test_format_catalog():
    catalog = [
        {
            "type": "ml.layers.conv2d",
            "displayName": "Conv2d",
            "category": ["ML", "Layers"],
            "description": "2D conv",
            "properties": [{"id": "outChannels", "kind": "number", "integer": True, "default": 64}],
            "ports": [
                {"id": "in", "direction": "input", "dataType": "tensor"},
                {"id": "out", "direction": "output", "dataType": "tensor", "allowMultiple": True},
            ],
        }
    ]
    text = format_catalog(catalog)
    assert "ml.layers.conv2d" in text
    assert "outChannels" in text
    assert "in:tensor" in text and "out:tensor" in text


def test_summarize_graph_forms():
    empty = summarize_graph({"version": "1.0", "graph": {"nodes": [], "edges": []}})
    assert "empty" in empty

    g = {
        "version": "1.0",
        "graph": {
            "name": "D",
            "nodes": [{"id": "c1", "type": "ml.layers.conv2d", "properties": {"outChannels": 32}}],
            "edges": [{"id": "e", "source": {"nodeId": "d", "portId": "out"}, "target": {"nodeId": "c1", "portId": "in"}}],
        },
    }
    s = summarize_graph(g)
    assert "c1 (ml.layers.conv2d)" in s
    assert "outChannels=32" in s
    assert "d.out -> c1.in" in s
    # Inner-graph form (no version wrapper) also works.
    assert "c1" in summarize_graph(g["graph"])


def test_run_turn_streams_and_stores_history(monkeypatch):
    class Stub(LLMProvider):
        name = "stub-test"

        async def stream_chat(self, *, system, messages):
            assert "NodeTorch Assistant" in system  # persona present
            assert "Conv2d" in system  # catalog present
            assert "Current graph" in messages[-1]["content"]  # graph in last user turn
            for word in ["Hello", " world"]:
                yield TextDelta(word)

    register("stub-test", Stub, label="stub", fields=[])
    monkeypatch.setattr(agent_mod, "load_config", lambda: AgentConfig(provider="stub-test", model="x"))

    reset_session("t")
    catalog = [{"type": "ml.layers.conv2d", "displayName": "Conv2d", "category": ["ML"], "properties": [], "ports": []}]
    graph = {"version": "1.0", "graph": {"nodes": [{"id": "c1", "type": "ml.layers.conv2d", "properties": {}}], "edges": []}}

    out: list[str] = []

    async def on_text(t):
        out.append(t)

    async def run():
        return await agent_mod.run_turn(
            session_id="t", message="hi", graph=graph, catalog=catalog, on_text=on_text
        )

    answer = asyncio.run(run())
    assert answer == "Hello world"
    assert "".join(out) == "Hello world"

    sess = get_session("t")
    assert len(sess.history) == 2
    assert sess.history[0]["role"] == "user" and sess.history[1]["role"] == "assistant"
    assert sess.catalog_text  # catalog cached on the session
