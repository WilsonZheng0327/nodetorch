import json
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agent.agent import run_turn, validate_provider
from agent.config import load_config, save_config
from agent.providers import list_providers
from agent.providers.base import ProviderError
from api._ws import ws_reader

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["agent"])


# ── AI agent ──────────────────────────────────────────────────────────────────
# Provider-agnostic chat assistant. The loop lives in backend/agent/. Config REST
# routes drive the settings UI; the /agent WebSocket streams answers.

@router.get("/agent/providers")
def agent_providers():
    """Registered providers + the config fields each needs (for the settings UI)."""
    return {"providers": list_providers()}


@router.get("/agent/config")
def agent_get_config():
    """Current provider/model/base_url — never the API key."""
    return load_config().public()


@router.post("/agent/config")
async def agent_set_config(body: dict):
    """Update provider/base_url/model/api_key. Returns the public view (no key)."""
    try:
        cfg = save_config(body)
        return {"status": "ok", "config": cfg.public()}
    except Exception as e:
        logger.error(f"Agent config save failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/agent/test")
async def agent_test():
    """Ping the configured provider to validate connectivity/credentials."""
    try:
        await validate_provider()
        return {"status": "ok"}
    except ProviderError as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        logger.error(f"Agent test failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


@router.websocket("/agent")
async def agent_websocket(ws: WebSocket):
    """Streaming chat. Mirrors /ws: a reader task feeds a queue so the loop can
    dispatch chat turns and cancels concurrently.

    Client → { type: 'chat', message, graph?, catalog?, sessionId? } | { type: 'cancel' }
    Server → { type: 'text_delta', text } … then { type: 'done' } | { type: 'error', error }
             | { type: 'cancelled' }
    """
    await ws.accept()
    logger.info("Agent WebSocket connected")

    msg_queue: asyncio.Queue = asyncio.Queue()
    reader_task = asyncio.create_task(ws_reader(ws, msg_queue))
    current_task: asyncio.Task | None = None

    async def send(obj: dict):
        await ws.send_text(json.dumps(obj))

    try:
        while True:
            msg = await msg_queue.get()
            mtype = msg.get("type")

            if mtype == "_disconnect":
                break

            if mtype == "cancel":
                if current_task and not current_task.done():
                    current_task.cancel()
                continue

            if mtype != "chat":
                continue

            # One turn at a time — cancel any still-running turn first.
            if current_task and not current_task.done():
                current_task.cancel()

            async def do_turn(msg=msg):
                async def on_text(text: str):
                    await send({"type": "text_delta", "text": text})
                try:
                    await run_turn(
                        session_id=msg.get("sessionId", "default"),
                        message=msg.get("message", ""),
                        graph=msg.get("graph"),
                        catalog=msg.get("catalog"),
                        on_text=on_text,
                    )
                    await send({"type": "done"})
                except asyncio.CancelledError:
                    try:
                        await send({"type": "cancelled"})
                    except Exception:
                        pass
                except ProviderError as e:
                    await send({"type": "error", "error": str(e)})
                except Exception as e:
                    logger.error(f"Agent turn failed: {e}", exc_info=True)
                    await send({"type": "error", "error": str(e)})

            current_task = asyncio.create_task(do_turn())

    except Exception as e:
        logger.error(f"Agent WebSocket error: {e}")
    finally:
        reader_task.cancel()
        if current_task and not current_task.done():
            current_task.cancel()
        logger.info("Agent WebSocket closed")
