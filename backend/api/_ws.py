import json
from fastapi import WebSocket, WebSocketDisconnect


async def ws_reader(ws: WebSocket, queue):
    try:
        while True:
            data = await ws.receive_text()
            await queue.put(json.loads(data))
    except WebSocketDisconnect:
        await queue.put({"type": "_disconnect"})
