import json
import asyncio
import threading
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from engine.graph_builder import execute_graph, train_graph
from api._ws import ws_reader

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["training"])


@router.post("/train")
async def train(graph_data: dict):
    logger.info("Training requested (REST)")
    try:
        results = train_graph(graph_data)
        if "error" in results:
            logger.error(f"Training failed: {results['error']}")
            return {"status": "error", "error": results["error"]}
        logger.info("Training complete")
        return {"status": "ok", "results": results}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {"status": "error", "error": str(e)}


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected")

    cancel_event: threading.Event | None = None

    msg_queue: asyncio.Queue = asyncio.Queue()
    reader_task = asyncio.create_task(ws_reader(ws, msg_queue))

    try:
        while True:
            msg = await msg_queue.get()

            if msg.get("type") == "_disconnect":
                break

            logger.info(f"WebSocket message: type={msg.get('type')}")

            if msg.get("type") == "forward":
                try:
                    results = execute_graph(msg["graph"])
                    await ws.send_text(json.dumps({
                        "type": "forward_result",
                        "status": "ok",
                        "results": results,
                    }))
                except Exception as e:
                    logger.error(f"WS forward failed: {e}")
                    await ws.send_text(json.dumps({
                        "type": "forward_result",
                        "status": "error",
                        "error": str(e),
                    }))

            elif msg.get("type") == "cancel":
                logger.info("Cancel requested")
                if cancel_event:
                    cancel_event.set()

            elif msg.get("type") == "train":
                logger.info("Training started via WebSocket")
                cancel_event = threading.Event()

                try:
                    epoch_queue: asyncio.Queue = asyncio.Queue()

                    def on_epoch(epoch_data):
                        logger.info(f"Epoch {epoch_data['epoch']}: loss={epoch_data['loss']:.4f}, acc={epoch_data['accuracy']:.4f}")
                        epoch_queue.put_nowait(("epoch", epoch_data))

                    def on_batch(batch_data):
                        epoch_queue.put_nowait(("batch", batch_data))

                    loop = asyncio.get_event_loop()
                    ce = cancel_event
                    train_task = loop.run_in_executor(
                        None, lambda: train_graph(msg["graph"], on_epoch=on_epoch, on_batch=on_batch, cancel_event=ce)
                    )

                    # Stream epoch/batch results while also checking for cancel via msg_queue
                    while not train_task.done():
                        try:
                            msg_type, data = await asyncio.wait_for(
                                epoch_queue.get(), timeout=0.3
                            )
                            await ws.send_text(json.dumps({
                                "type": msg_type,
                                **data,
                            }))
                        except asyncio.TimeoutError:
                            pass

                        # Check for cancel messages that arrived
                        while not msg_queue.empty():
                            pending = msg_queue.get_nowait()
                            if pending.get("type") == "cancel":
                                logger.info("Cancel requested during training")
                                cancel_event.set()
                            elif pending.get("type") == "_disconnect":
                                logger.info("Disconnect during training, cancelling")
                                cancel_event.set()

                    # Drain remaining results
                    while not epoch_queue.empty():
                        msg_type, data = epoch_queue.get_nowait()
                        await ws.send_text(json.dumps({
                            "type": msg_type,
                            **data,
                        }))

                    results = train_task.result()
                    cancelled = cancel_event.is_set()

                    if "error" in results:
                        logger.error(f"Training failed: {results['error']}")
                        await ws.send_text(json.dumps({
                            "type": "train_result",
                            "status": "error",
                            "error": results["error"],
                        }))
                    else:
                        status = "cancelled" if cancelled else "ok"
                        logger.info(f"Training {'cancelled' if cancelled else 'complete'}")
                        await ws.send_text(json.dumps({
                            "type": "train_result",
                            "status": "ok",
                            "results": results,
                            "cancelled": cancelled,
                        }))

                    cancel_event = None

                except Exception as e:
                    logger.error(f"WS training failed: {e}", exc_info=True)
                    try:
                        await ws.send_text(json.dumps({
                            "type": "train_result",
                            "status": "error",
                            "error": str(e),
                        }))
                    except Exception:
                        pass

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        reader_task.cancel()
        if cancel_event:
            cancel_event.set()
        logger.info("WebSocket closed")
