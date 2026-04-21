# main.py — FastAPI backend for NodeTorch.
#
# Endpoints:
#   GET  /health              — server health check
#   POST /forward             — single forward pass, returns per-node results
#   POST /train               — synchronous training (REST, no streaming)
#   POST /infer               — inference using stored trained weights
#   GET  /dataset/{type}      — dataset detail info (labels, sample images)
#   GET  /blocks              — list saved + preset block templates
#   POST /blocks/save         — save a custom block as JSON
#   GET  /blocks/{filename}   — load a specific block
#   DELETE /blocks/{filename} — delete a user-saved block
#   WS   /ws                  — WebSocket for streaming training (epoch results in real-time)
#
# WebSocket protocol:
#   Client sends: { type: "train", graph: {...} } or { type: "cancel" }
#   Server sends: { type: "epoch", epoch, loss, accuracy, time, ... } per epoch
#                 { type: "train_result", status, results, cancelled? } when done
#
# Training runs in a thread executor so the event loop stays responsive.
# A separate reader task reads WebSocket messages concurrently with epoch streaming,
# enabling cancel messages to be processed during training.

import warnings
warnings.filterwarnings("ignore", message="dtype.*align")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import threading
import logging

from graph_builder import execute_graph, train_graph, infer_graph, evaluate_test_set, get_layer_detail, get_device_name, set_device, save_model, load_model, save_model_bytes, load_model_bytes
from export_python import export_to_python
from step_through import run_step_through
from activation_max import activation_maximization
from backprop_sim import simulate_backprop, run_backward_step_through
from loss_landscape import compute_loss_landscape
from latent_viz import generate_latent_grid
from denoise_viz import run_denoise_step_through
from gan_generate import generate_gan_images
from text_generate import generate_text
from runs_store import list_runs, load_run, delete_run
from data_loaders import DATASET_DETAILS, augmentation_preview
import os
from pathlib import Path

STORAGE_DIR = Path("./storage")
STORAGE_DIR.mkdir(exist_ok=True)
BLOCKS_DIR = STORAGE_DIR / "blocks"
BLOCKS_DIR.mkdir(exist_ok=True)
PRESETS_DIR = STORAGE_DIR / "presets"  # shipped block templates, read-only

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


@app.get("/system-info")
def system_info():
    """Return system GPU/device information."""
    import torch
    import platform
    info: dict = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cudaAvailable": torch.cuda.is_available(),
        "gpuCount": 0,
        "gpus": [],
    }
    if torch.cuda.is_available():
        info["gpuCount"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "name": props.name,
                "vram": round(props.total_memory / 1024**3, 1),
                "computeCapability": f"{props.major}.{props.minor}",
            })
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info["mpsAvailable"] = True
    info["currentDevice"] = get_device_name()
    return info


@app.post("/set-device")
async def set_device_endpoint(request: dict):
    """Set the training device (cpu, cuda, cuda:0, mps)."""
    device = request.get("device", "cpu")
    logger.info(f"Device set to: {device}")
    try:
        import torch
        # Validate the device exists
        torch.device(device)
        set_device(device)
        return {"status": "ok", "device": device}
    except Exception as e:
        return {"status": "error", "error": str(e)}


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


@app.post("/evaluate-test")
async def evaluate_test(request: dict):
    """Evaluate trained model on the held-out test set."""
    logger.info("Test set evaluation requested")
    try:
        result = evaluate_test_set(request["graph"])
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Test evaluation failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/gan-generate")
async def gan_generate(request: dict):
    """Generate images from a trained GAN."""
    logger.info("GAN generate requested")
    try:
        result = generate_gan_images(
            request["graph"],
            num_samples=request.get("numSamples", 8),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"GAN generate failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/denoise-step-through")
async def denoise_step_through(request: dict):
    """Run DDPM denoising and return images at each timestep."""
    logger.info("Denoise step-through requested")
    try:
        result = run_denoise_step_through(
            request["graph"],
            num_samples=request.get("numSamples", 4),
            capture_every=request.get("captureEvery", 1),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Denoise step-through failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/latent-grid")
async def latent_grid(request: dict):
    """Generate a latent space interpolation grid for a trained VAE."""
    logger.info("Latent grid requested")
    try:
        result = generate_latent_grid(
            request["graph"],
            grid_size=request.get("gridSize", 10),
            latent_range=request.get("latentRange", 3.0),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Latent grid failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/generate-text")
async def generate_text_endpoint(request: dict):
    """Generate text autoregressively from a trained language model."""
    logger.info("Text generation requested")
    try:
        result = generate_text(
            request["graph"],
            prompt=request.get("prompt", ""),
            max_tokens=request.get("maxTokens", 200),
            temperature=request.get("temperature", 0.8),
            top_k=request.get("topK", 0),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
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


@app.post("/save-model")
async def save_model_endpoint(request: dict):
    """Save trained model weights to disk."""
    logger.info("Save model requested")
    filepath = request.get("filepath", "saved_models/current.pt")
    try:
        result = save_model(filepath)
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return result
    except Exception as e:
        logger.error(f"Save model failed: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/download-weights")
async def download_weights():
    """Download trained weights as a .pt file."""
    from fastapi.responses import Response
    data = save_model_bytes()
    if data is None:
        return {"status": "error", "error": "No trained model to save"}
    return Response(content=data, media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=weights.pt"})


@app.post("/upload-weights")
async def upload_weights(file: UploadFile, graph: str = Form(...)):
    """Upload a .pt file and load weights into the current graph."""
    logger.info("Upload weights requested")
    try:
        data = await file.read()
        graph_data = json.loads(graph)
        result = load_model_bytes(graph_data, data)
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return result
    except Exception as e:
        logger.error(f"Upload weights failed: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/saved-models")
async def list_saved_models():
    """List saved weight files."""
    model_dir = STORAGE_DIR / "weights"
    if not model_dir.exists():
        return {"status": "ok", "files": []}
    files = sorted([f.name for f in model_dir.glob("*.pt")])
    return {"status": "ok", "files": files}


@app.post("/load-model")
async def load_model_endpoint(request: dict):
    """Load trained model weights from disk."""
    logger.info("Load model requested")
    filepath = request.get("filepath", "saved_models/current.pt")
    try:
        result = load_model(request["graph"], filepath)
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return result
    except Exception as e:
        logger.error(f"Load model failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/step-through")
async def step_through(request: dict):
    """Run a forward pass through the graph on a single sample, returning ordered stages.

    Request: { graph, sampleIdx? }
    Response: { status: "ok", result: { stages, sample } } or error
    """
    logger.info("Step-through requested")
    try:
        result = run_step_through(
            request["graph"],
            request.get("sampleIdx"),
            mask=request.get("mask"),
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Step-through failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/augmentation-preview")
async def aug_preview(request: dict):
    """Preview augmentation effects on a sample."""
    try:
        result = augmentation_preview(
            request["datasetType"],
            augHFlip=request.get("augHFlip", False),
            augRandomCrop=request.get("augRandomCrop", False),
            augColorJitter=request.get("augColorJitter", False),
        )
        if result and "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Augmentation preview failed: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/runs")
async def get_runs():
    """List all saved training runs (summary only)."""
    return {"status": "ok", "runs": list_runs()}


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Load a full training run including per-epoch history."""
    data = load_run(run_id)
    if data is None:
        return {"status": "error", "error": "Run not found"}
    return {"status": "ok", "run": data}


@app.delete("/runs/{run_id}")
async def remove_run(run_id: str):
    ok = delete_run(run_id)
    return {"status": "ok" if ok else "error"}


@app.post("/loss-landscape")
async def loss_landscape(request: dict):
    """Compute 2D loss surface around current weights."""
    logger.info("Loss landscape requested")
    try:
        result = compute_loss_landscape(
            request["graph"],
            grid_size=request.get("gridSize", 11),
            alpha_range=request.get("alphaRange", 1.0),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Loss landscape failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/simulate-backprop")
async def sim_backprop(request: dict):
    """Run one forward+backward pass and return per-node gradient magnitudes."""
    logger.info("Backprop simulation requested")
    try:
        result = simulate_backprop(request["graph"])
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Backprop simulation failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/backward-step-through")
async def backward_step_through(request: dict):
    """Run a forward+backward pass and return rich per-node backward stages."""
    logger.info("Backward step-through requested")
    try:
        result = run_backward_step_through(request["graph"])
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Backward step-through failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/activation-max")
async def activation_max(request: dict):
    """Run activation maximization on a Conv2d node — find input images that maximize each filter."""
    logger.info(f"Activation max for node {request.get('nodeId')}")
    try:
        result = activation_maximization(
            request["graph"],
            request["nodeId"],
            num_filters=request.get("numFilters", 8),
            iterations=request.get("iterations", 25),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Activation max failed: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/layer-detail")
async def layer_detail(request: dict):
    """Return detailed visualization data for a specific node (weight matrix, feature maps, etc.)."""
    logger.info(f"Layer detail requested for node {request.get('nodeId')}")
    try:
        result = get_layer_detail(request["graph"], request["nodeId"])
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "detail": result}
    except Exception as e:
        logger.error(f"Layer detail failed: {e}")
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


@app.get("/presets")
async def list_presets():
    import glob
    preset_dir = Path(__file__).resolve().parent.parent / "model-presets"
    presets = []
    for f in sorted(preset_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            presets.append({"filename": f.name, "name": data.get("graph", {}).get("name", f.stem)})
        except Exception:
            continue
    return {"status": "ok", "presets": presets}


@app.post("/presets/load")
async def load_preset(request: dict):
    filepath = Path(__file__).resolve().parent.parent / "model-presets" / request["filename"]
    if not filepath.exists():
        return {"status": "error", "error": "Preset not found"}
    return {"status": "ok", "data": json.loads(filepath.read_text())}


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


@app.post("/export-python")
async def export_python_endpoint(request: dict):
    """Generate a standalone Python training script from the graph."""
    from fastapi.responses import Response
    logger.info("Export Python requested")
    try:
        code = export_to_python(request["graph"])
        graph_name = request["graph"].get("graph", {}).get("name", "model").replace(" ", "_").lower()
        # HTTP headers must be ASCII/latin-1 — strip non-ASCII from filename
        graph_name = graph_name.encode("ascii", "ignore").decode("ascii") or "model"
        return Response(
            content=code.encode("utf-8"),
            media_type="text/x-python; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{graph_name}.py"'}
        )
    except Exception as e:
        logger.error(f"Export Python failed: {e}")
        return {"status": "error", "error": str(e)}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
