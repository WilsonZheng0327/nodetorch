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
from agent.providers.base import LLMProvider, ProviderError
from agent.providers.registry import register
from agent.session import get_session, reset_session
from agent.tools import GRAPH_TOOLS


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


def test_graph_tools_defined():
    names = {t.name for t in GRAPH_TOOLS}
    assert names == {
        "set_node_property", "add_node", "connect", "remove_node", "remove_edge",
        "clear_graph", "organize_layout", "add_block",
        "enter_block", "exit_block", "save_block",
        "get_graph", "get_node", "get_training_results", "validate",
    }
    for t in GRAPH_TOOLS:
        assert t.parameters["type"] == "object"  # valid JSON-schema shape


def test_run_turn_explain_only(monkeypatch):
    class Stub(LLMProvider):
        name = "stub-explain"

        async def run(self, *, system, messages, on_text, tools=None, execute_tool=None):
            assert "NodeTorch Assistant" in system  # persona present
            assert "Conv2d" in system  # catalog present
            assert "Current graph" in messages[-1]["content"]  # graph in last user turn
            assert not tools  # no execute_tool given → explain-only, no tools
            for word in ["Hello", " world"]:
                await on_text(word)

    register("stub-explain", Stub, label="stub", fields=[])
    monkeypatch.setattr(agent_mod, "load_config", lambda: AgentConfig(provider="stub-explain", model="x"))

    reset_session("t")
    catalog = [{"type": "ml.layers.conv2d", "displayName": "Conv2d", "category": ["ML"], "properties": [], "ports": []}]
    graph = {"version": "1.0", "graph": {"nodes": [{"id": "c1", "type": "ml.layers.conv2d", "properties": {}}], "edges": []}}

    out: list[str] = []

    async def on_text(t):
        out.append(t)

    answer = asyncio.run(
        agent_mod.run_turn(session_id="t", message="hi", graph=graph, catalog=catalog, on_text=on_text)
    )
    assert answer == "Hello world" == "".join(out)

    sess = get_session("t")
    assert len(sess.history) == 2
    assert sess.history[0]["role"] == "user" and sess.history[1]["role"] == "assistant"
    assert sess.catalog_text  # catalog cached on the session


def test_openai_compat_retries_malformed_tool_call():
    """A 'Failed to call a function' error is nudged + retried, not surfaced raw."""
    from types import SimpleNamespace as NS

    from openai import OpenAIError
    from agent.providers.openai_compat import OpenAICompatProvider

    def text_chunk(t):
        return NS(choices=[NS(delta=NS(content=t, tool_calls=None))])

    class FakeStream:
        def __init__(self, chunks):
            self._chunks = list(chunks)

        def __aiter__(self):
            self._it = iter(self._chunks)
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **kw):
            self.calls += 1
            if self.calls == 1:
                raise OpenAIError("Failed to call a function. Please adjust your prompt.")
            return FakeStream([text_chunk("Recovered "), text_chunk("answer.")])

    fake = NS(chat=NS(completions=FakeCompletions()))

    async def _close():
        pass

    fake.close = _close

    prov = OpenAICompatProvider(AgentConfig(provider="openai_compat", model="x"))
    prov._client = lambda: fake

    out: list[str] = []

    async def on_text(t):
        out.append(t)

    asyncio.run(prov.run(system="s", messages=[{"role": "user", "content": "hi"}], on_text=on_text, tools=GRAPH_TOOLS, execute_tool=lambda n, a: None))
    assert "".join(out) == "Recovered answer."  # recovered after the retry
    assert fake.chat.completions.calls == 2  # one failed, one succeeded


def test_run_turn_tool_loop(monkeypatch):
    """With an execute_tool bridge, tools are passed and the loop calls them."""

    class StubTool(LLMProvider):
        name = "stub-tool"

        async def run(self, *, system, messages, on_text, tools=None, execute_tool=None):
            assert any(t.name == "set_node_property" for t in (tools or []))  # tools passed
            assert "EDIT the graph" in system  # editing persona active
            await on_text("editing… ")
            result = await execute_tool("set_node_property", {"nodeId": "c1", "key": "outChannels", "value": 64})
            await on_text(f"({result})")

    register("stub-tool", StubTool, label="stub", fields=[])
    monkeypatch.setattr(agent_mod, "load_config", lambda: AgentConfig(provider="stub-tool", model="x"))

    reset_session("e")
    catalog = [{"type": "ml.layers.conv2d", "displayName": "Conv2d", "category": ["ML"], "properties": [], "ports": []}]
    graph = {"version": "1.0", "graph": {"nodes": [{"id": "c1", "type": "ml.layers.conv2d", "properties": {}}], "edges": []}}

    calls: list = []

    async def on_text(_t):
        pass

    async def execute_tool(name, args):
        calls.append((name, args))
        return "ok: set c1.outChannels=64"

    answer = asyncio.run(
        agent_mod.run_turn(
            session_id="e", message="set channels to 64", graph=graph, catalog=catalog,
            on_text=on_text, execute_tool=execute_tool,
        )
    )
    assert calls == [("set_node_property", {"nodeId": "c1", "key": "outChannels", "value": 64})]
    assert "ok: set c1.outChannels=64" in answer


def _run_build_turn(monkeypatch, *, build_calls, validate_results, repair_calls=None):
    """Run a turn whose stub provider builds, then auto-repairs, returning (calls, answer).

    The stub runs `build_calls` on its first invocation and `repair_calls` on each
    later (repair-round) invocation. `validate_results` is a list of strings the
    successive validate() calls return (the last one repeats if exhausted), so a
    test can simulate "fails, then passes". execute_tool records every call.
    """
    runs = {"n": 0}

    class StubBuild(LLMProvider):
        name = "stub-build"

        async def run(self, *, system, messages, on_text, tools=None, execute_tool=None):
            runs["n"] += 1
            calls_this_round = build_calls if runs["n"] == 1 else (repair_calls or [])
            await on_text("…")
            for name, args in calls_this_round:
                await execute_tool(name, args)

    register("stub-build", StubBuild, label="stub", fields=[])
    monkeypatch.setattr(agent_mod, "load_config", lambda: AgentConfig(provider="stub-build", model="x"))

    reset_session("b")
    catalog = [{"type": "ml.layers.conv2d", "displayName": "Conv2d", "category": ["ML"], "properties": [], "ports": []}]
    graph = {"version": "1.0", "graph": {"nodes": [], "edges": []}}

    calls: list = []
    vresults = iter(validate_results)

    async def on_text(_t):
        pass

    async def execute_tool(name, args):
        calls.append((name, args))
        if name == "validate":
            return next(vresults, validate_results[-1])
        return f"ok: {name}"

    answer = asyncio.run(
        agent_mod.run_turn(
            session_id="b", message="build a conv", graph=graph, catalog=catalog,
            on_text=on_text, execute_tool=execute_tool,
        )
    )
    return calls, answer


def test_finish_pass_validates_then_tidies_after_a_clean_build(monkeypatch):
    """A constructive turn that validates clean ends with validate then organize_layout."""
    calls, answer = _run_build_turn(
        monkeypatch,
        build_calls=[("add_node", {"type": "ml.layers.conv2d", "id": "c2"})],
        validate_results=["ok: no forward validation issues"],
    )
    names = [n for n, _ in calls]
    assert names == ["add_node", "validate", "organize_layout"]
    assert ("validate", {"mode": "forward"}) in calls
    assert "⚠️" not in answer  # clean validation → no warning, no repair round


def test_finish_pass_auto_repairs_then_converges(monkeypatch):
    """A failing validation is fed back; the repair round fixes it and no warning shows."""
    calls, answer = _run_build_turn(
        monkeypatch,
        build_calls=[("add_node", {"type": "ml.layers.conv2d", "id": "c2"})],
        validate_results=["1 forward issue(s):\n- c2: input not connected", "ok: clean"],
        repair_calls=[("connect", {"sourceId": "d", "sourcePort": "out", "targetId": "c2", "targetPort": "in"})],
    )
    names = [n for n, _ in calls]
    # build → validate(fail) → repair connect → validate(ok) → organize
    assert names == ["add_node", "validate", "connect", "validate", "organize_layout"]
    assert "⚠️" not in answer  # converged within the budget


def test_finish_pass_gives_up_after_repair_budget_and_reports(monkeypatch):
    """If repairs never fix it, stop after _MAX_REPAIR_ROUNDS and surface the issue."""
    calls, answer = _run_build_turn(
        monkeypatch,
        build_calls=[("connect", {"sourceId": "a", "sourcePort": "out", "targetId": "b", "targetPort": "in"})],
        validate_results=["1 forward issue(s):\n- c2: input not connected"],  # always fails
        repair_calls=[("get_node", {"nodeId": "c2"})],  # model tries but can't fix
    )
    # One initial validate + one per repair round = 1 + _MAX_REPAIR_ROUNDS validates.
    assert [n for n, _ in calls].count("validate") == 1 + agent_mod._MAX_REPAIR_ROUNDS
    assert [n for n, _ in calls].count("get_node") == agent_mod._MAX_REPAIR_ROUNDS
    assert "⚠️" in answer and "input not connected" in answer


def test_no_finish_pass_without_a_build(monkeypatch):
    """A property-only turn must NOT trigger validate/organize (don't disturb layout)."""
    calls, _ = _run_build_turn(
        monkeypatch,
        build_calls=[("set_node_property", {"nodeId": "c1", "key": "outChannels", "value": 64})],
        validate_results=["ok"],
    )
    names = [n for n, _ in calls]
    assert names == ["set_node_property"]  # no validate / organize_layout appended
