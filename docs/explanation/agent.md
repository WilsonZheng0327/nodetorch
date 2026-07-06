# AI Assistant Architecture

NodeTorch ships an integrated, **provider-agnostic** AI assistant. It can read
the current graph, explain it, and edit it (add/remove/connect nodes) by calling
tools the browser applies locally. The agent loop runs in the **backend**; the
browser hosts the chat UI and a tool-bridge.

```
ChatRail (browser)                          backend/agent/  (FastAPI)
  useAgentChat ── WS /agent ──▶  routes.py  ──▶  agent.py (loop)
   { chat, message, graph,                        context.py  (build messages)
     catalog, sessionId }                         catalog.py  (format node catalog)
   ◀── { text_delta }                             session.py  (per-session history)
   ◀── { tool_call } ──▶ graphTools (apply)       providers/  (registry → adapter)
   ── { tool_result } ──▶                          openai_compat | anthropic
   ◀── { done | error | cancelled }                     │ httpx
                                                  paid API key OR local model (Ollama, …)
```

## Backend — `backend/agent/`

- **`agent.py`** — the loop: build messages → stream a completion → if the model
  emits a tool call, dispatch it, feed the result back, and continue.
- **`providers/`** — a registry (`registry.py`: `register()` / `get()`) mapping a
  provider name to an adapter. `openai_compat.py` reaches OpenAI, OpenRouter,
  Groq, and local servers (Ollama/LM Studio) via a configurable `base_url`;
  `anthropic.py` is the native Claude adapter. Adding a provider = one adapter +
  one `register()` call.
- **`config.py`** — provider selection + per-provider config (provider, base_url,
  model, api_key). Loaded from env vars and/or a gitignored
  `backend/storage/agent_config.json`. **API keys never leave the backend** — only
  provider/model/base_url metadata is exposed to the browser.
- **`context.py`** — assembles each turn within a token budget: system prompt
  (persona + formatted node catalog) + a compact graph representation + recent
  history.
- **`catalog.py`** — formats the frontend-supplied node catalog into prompt text.
- **`session.py`** — in-memory conversation history keyed by `sessionId`.
- **`tools.py`** — the tool schemas the model may call.
- **`routes.py`** — the WebSocket + config REST routes (below).

## Transport (`routes.py`)

**WebSocket `/agent`:**

- Client → `{ type: 'chat', message, graph, catalog, sessionId }`,
  `{ type: 'tool_result', id, result }`, `{ type: 'cancel' }`.
- Server → `{ type: 'text_delta', text }`,
  `{ type: 'tool_call', id, name, args }` (the browser applies it and replies with
  `tool_result`), then `{ type: 'done' }` | `{ type: 'error', error }` |
  `{ type: 'cancelled' }`.

**Config REST:** `GET /agent/providers` (registered providers + config fields),
`GET /agent/config` (current provider/model/base_url — **no key**),
`POST /agent/config` (set provider/base_url/model/api_key),
`POST /agent/test` (validate the configured provider with a tiny ping).

## Frontend — `src/ui/chat/`

- **`ChatRail.tsx`** — the docked panel: message list (markdown), input, stop
  button, settings. Toggled with the `3` shortcut.
- **`useAgentChat.ts`** — owns the `/agent` WebSocket: streaming assembly, session
  id, and the **tool-bridge** — on a `tool_call` it runs the tool locally and
  replies with a `tool_result`. Sends the node catalog once per connection.
- **`graphTools.ts`** — the tool executor. Applies the model's graph edits through
  `useGraph` (guarded by the same `isValidConnection` / validation rules a human
  action goes through) and answers read tools (e.g. training history).
- **`AgentSettings.tsx`** — provider dropdown + base URL + model + API key, backed
  by the config REST routes; "Test connection" pings `POST /agent/test`.

## The node catalog

The rich node catalog (types, properties, ports, descriptions) lives only in the
**frontend** registry. `src/domain/catalog.ts` (`buildNodeCatalog`) builds it from
`domain.nodeRegistry`, and `useAgentChat` sends it to the backend once per
connection (cached server-side). New nodes therefore appear to the agent
automatically — no second, drift-prone catalog on the backend.

## Enabling it

The assistant needs a provider configured (see the Setup section in the README).
For a paid API set a key via the in-app **Agent settings** (stored server-side);
for a local model, point the base URL at e.g. Ollama (`http://localhost:11434/v1`)
with no key. `requirements.txt` installs `openai` and `anthropic` for the adapters.
