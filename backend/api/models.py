import json
import logging
from fastapi import APIRouter, UploadFile, Form

from engine.graph_builder import (
    save_model, load_model, save_model_bytes, load_model_bytes,
    save_bundle_bytes, load_bundle_bytes,
)
from paths import STORAGE_DIR

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["models"])


@router.post("/save-model")
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


@router.post("/load-model")
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


@router.get("/saved-models")
async def list_saved_models():
    """List saved weight files."""
    model_dir = STORAGE_DIR / "weights"
    if not model_dir.exists():
        return {"status": "ok", "files": []}
    files = sorted([f.name for f in model_dir.glob("*.pt")])
    return {"status": "ok", "files": files}


@router.get("/download-weights")
async def download_weights():
    """Download trained weights as a .pt file."""
    from fastapi.responses import Response
    data = save_model_bytes()
    if data is None:
        return {"status": "error", "error": "No trained model to save"}
    return Response(content=data, media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=weights.pt"})


@router.post("/upload-weights")
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


@router.post("/download-model")
async def download_model(request: dict):
    """Download a self-contained model bundle (graph + weights) as a .ntmodel file.
    The frontend sends the current graph; the backend pairs it with the trained
    weights so the bundle reloads on its own."""
    from fastapi.responses import Response
    graph_data = request.get("graph")
    if graph_data is None:
        return {"status": "error", "error": "Missing graph"}
    data = save_bundle_bytes(graph_data)
    if data is None:
        return {"status": "error", "error": "No trained model to save"}
    return Response(content=data, media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=model.ntmodel"})


@router.post("/upload-model")
async def upload_model(file: UploadFile):
    """Upload a .ntmodel bundle, load its weights, and return the embedded graph so
    the frontend can put the architecture on the canvas. Self-contained — no graph
    needs to be sent."""
    logger.info("Upload model requested")
    try:
        result = load_bundle_bytes(await file.read())
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return result  # { status: "ok", graph: <SerializedGraph> }
    except Exception as e:
        logger.error(f"Upload model failed: {e}")
        return {"status": "error", "error": str(e)}
