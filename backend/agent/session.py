"""In-memory conversation sessions, keyed by a client-supplied sessionId.

Mirrors the single-process global-state style used elsewhere in the backend
(e.g. the model store). Holds the message history and the formatted node catalog
(cached so the frontend only sends it once per session).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Keep the prompt bounded — last N messages (user+assistant) carried forward.
MAX_HISTORY_MESSAGES = 12


@dataclass
class AgentSession:
    session_id: str
    history: list[dict] = field(default_factory=list)  # [{"role","content"}]
    catalog_text: str = ""
    blocks_text: str = ""

    def add_user(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self) -> None:
        if len(self.history) > MAX_HISTORY_MESSAGES:
            # Drop oldest, but keep the history starting on a user turn.
            self.history = self.history[-MAX_HISTORY_MESSAGES:]
            while self.history and self.history[0]["role"] != "user":
                self.history.pop(0)


_sessions: dict[str, AgentSession] = {}


def get_session(session_id: str) -> AgentSession:
    sess = _sessions.get(session_id)
    if sess is None:
        sess = AgentSession(session_id=session_id)
        _sessions[session_id] = sess
    return sess


def reset_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
