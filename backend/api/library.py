import json
import logging
from pathlib import Path
from fastapi import APIRouter

from paths import BLOCKS_DIR, PRESETS_DIR

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["library"])

# Repo-root model-presets dir. main.py lived at backend/main.py, so its
# Path(__file__).resolve().parent.parent pointed at the repo root. This module
# is one directory deeper (backend/api/), so go up one extra level to resolve
# to the same location.
_MODEL_PRESETS_DIR = Path(__file__).resolve().parent.parent.parent / "model-presets"


@router.get("/presets")
async def list_presets():
    import glob
    preset_dir = _MODEL_PRESETS_DIR
    presets = []
    for f in sorted(preset_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            presets.append({"filename": f.name, "name": data.get("graph", {}).get("name", f.stem)})
        except Exception:
            continue
    return {"status": "ok", "presets": presets}


@router.post("/presets/load")
async def load_preset(request: dict):
    filepath = _MODEL_PRESETS_DIR / request["filename"]
    if not filepath.exists():
        return {"status": "error", "error": "Preset not found"}
    return {"status": "ok", "data": json.loads(filepath.read_text())}


@router.get("/blocks")
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


@router.post("/blocks/save")
async def save_block(block_data: dict):
    """Save a block definition to the blocks directory."""
    name = block_data.get("name", "untitled")
    filename = name.lower().replace(" ", "-") + ".json"
    filepath = BLOCKS_DIR / filename
    filepath.write_text(json.dumps(block_data, indent=2))
    logger.info(f"Block saved: {filepath}")
    return {"status": "ok", "filename": filename}


@router.get("/blocks/{filename:path}")
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


@router.delete("/blocks/{filename:path}")
async def delete_block(filename: str):
    """Delete a saved block (presets cannot be deleted)."""
    if filename.startswith("preset:"):
        return {"status": "error", "error": "Cannot delete preset blocks"}
    filepath = BLOCKS_DIR / filename
    if filepath.exists():
        filepath.unlink()
        logger.info(f"Block deleted: {filepath}")
    return {"status": "ok"}
