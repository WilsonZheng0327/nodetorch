"""Provider-agnostic AI agent for NodeTorch.

The agent loop runs server-side and streams answers to the chat panel. It is not
tied to any single LLM provider — see `providers/` for the pluggable adapter
registry (OpenAI-compatible covers paid APIs and local models; Anthropic native;
add more by registering an adapter).

v1 is explain-only: it answers questions about the user's graph. The loop and
WebSocket protocol are structured to extend to tool-use (acting on the graph)
later without rework.
"""
