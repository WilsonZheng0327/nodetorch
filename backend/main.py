# FastAPI backend for NodeTorch.
# Receives serialized graphs, runs PyTorch forward passes and training, returns results.

import warnings
warnings.filterwarnings("ignore", message="dtype.*align")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import threading
import logging

from graph_builder import execute_graph, train_graph, infer_graph
from data_loaders import DATASET_DETAILS
import os
from pathlib import Path

BLOCKS_DIR = Path("./blocks")
BLOCKS_DIR.mkdir(exist_ok=True)
PRESETS_DIR = Path("./presets")  # shipped block templates, read-only

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nodetorch")

app = FastAPI(title="NodeTorch Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forward")
async def forward(graph_data: dict):
    logger.info("Forward pass requested")
    try:
        results = execute_graph(graph_data)
        logger.info("Forward pass complete")
        return {"status": "ok", "results": results}
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/train")
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


@app.post("/infer")
async def infer(graph_data: dict):
    """Run inference using trained weights on a single sample."""
    logger.info("Inference requested")
    try:
        results = infer_graph(graph_data)
        if "error" in results:
            logger.error(f"Inference failed: {results['error']}")
            return {"status": "error", "error": results["error"]}
        logger.info(f"Inference complete: prediction={results.get('prediction')}")
        return {"status": "ok", "results": results}
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/dataset/{dataset_type:path}")
async def dataset_detail(dataset_type: str):
    """Return detailed info about a dataset (labels, sample images, stats)."""
    logger.info(f"Dataset detail requested: {dataset_type}")
    detail_fn = DATASET_DETAILS.get(dataset_type)
    if not detail_fn:
        return {"status": "error", "error": f"Unknown dataset: {dataset_type}"}
    try:
        detail = detail_fn()
        return {"status": "ok", "detail": detail}
    except Exception as e:
        logger.error(f"Dataset detail failed: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/blocks")
async def list_blocks():
    """List all saved and preset blocks."""
    blocks = []
    # Preset blocks (shipped, read-only)
    if PRESETS_DIR.exists():
        for f in PRESETS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                blocks.append({
                    "filename": f"preset:{f.name}",
                    "name": data.get("name", f.stem),
                    "description": data.get("description", ""),
                    "preset": True,
                })
            except Exception:
                continue
    # User-saved blocks
    for f in BLOCKS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            blocks.append({
                "filename": f.name,
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "preset": False,
            })
        except Exception:
            continue
    return {"status": "ok", "blocks": blocks}


@app.post("/blocks/save")
async def save_block(block_data: dict):
    """Save a block definition to the blocks directory."""
    name = block_data.get("name", "untitled")
    filename = name.lower().replace(" ", "-") + ".json"
    filepath = BLOCKS_DIR / filename
    filepath.write_text(json.dumps(block_data, indent=2))
    logger.info(f"Block saved: {filepath}")
    return {"status": "ok", "filename": filename}


@app.get("/blocks/{filename:path}")
async def load_block(filename: str):
    """Load a block definition (from presets or user blocks)."""
    if filename.startswith("preset:"):
        filepath = PRESETS_DIR / filename[7:]
    else:
        filepath = BLOCKS_DIR / filename
    if not filepath.exists():
        return {"status": "error", "error": f"Block not found: {filename}"}
    data = json.loads(filepath.read_text())
    return {"status": "ok", "block": data}


@app.delete("/blocks/{filename:path}")
async def delete_block(filename: str):
    """Delete a saved block (presets cannot be deleted)."""
    if filename.startswith("preset:"):
        return {"status": "error", "error": "Cannot delete preset blocks"}
    filepath = BLOCKS_DIR / filename
    if filepath.exists():
        filepath.unlink()
        logger.info(f"Block deleted: {filepath}")
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected")

    cancel_event: threading.Event | None = None

    async def read_messages(msg_queue: asyncio.Queue):
        """Read WebSocket messages and put them in a queue."""
        try:
            while True:
                data = await ws.receive_text()
                await msg_queue.put(json.loads(data))
        except WebSocketDisconnect:
            await msg_queue.put({"type": "_disconnect"})

    msg_queue: asyncio.Queue = asyncio.Queue()
    reader_task = asyncio.create_task(read_messages(msg_queue))

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
                        epoch_queue.put_nowait(epoch_data)

                    loop = asyncio.get_event_loop()
                    ce = cancel_event
                    train_task = loop.run_in_executor(
                        None, lambda: train_graph(msg["graph"], on_epoch=on_epoch, cancel_event=ce)
                    )

                    # Stream epoch results while also checking for cancel via msg_queue
                    while not train_task.done():
                        # Check for epoch data
                        try:
                            epoch_data = await asyncio.wait_for(
                                epoch_queue.get(), timeout=0.3
                            )
                            await ws.send_text(json.dumps({
                                "type": "epoch",
                                **epoch_data,
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

                    # Drain remaining epoch results
                    while not epoch_queue.empty():
                        epoch_data = epoch_queue.get_nowait()
                        await ws.send_text(json.dumps({
                            "type": "epoch",
                            **epoch_data,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
