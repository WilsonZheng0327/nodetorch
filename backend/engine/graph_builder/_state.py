"""Device selection and the in-memory trained-model store.

Holds the engine's shared mutable state: the active device, the trained-module
store (``_model_store``), and the last-run cache (``_last_run``). Other engine
submodules mutate ``_model_store`` / ``_last_run`` in place, so these dict
objects must stay singletons — the package re-exports the same references.
"""

import os
import torch
import torch.nn as nn


# --- Device management ---
def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

_device: str = _default_device()

def get_device() -> torch.device:
    return torch.device(_device)

def set_device(device: str):
    global _device
    _device = device

def get_device_name() -> str:
    return _device

# Stores trained modules in memory so inference can reuse them.
# Key: "current" (single session for now), Value: dict of node_id → nn.Module
_model_store: dict[str, dict[str, nn.Module]] = {}

# Cache of the last forward/train/infer pass for layer detail queries.
# Avoids re-running the graph when the user opens the detail modal.
_last_run: dict = {}

def has_trained_model() -> bool:
    return "current" in _model_store

def get_trained_modules() -> dict[str, nn.Module]:
    return _model_store.get("current", {})


def save_model(filepath: str = "storage/weights/current.pt") -> dict:
    """Save trained module state dicts to disk."""
    if "current" not in _model_store:
        return {"error": "No trained model to save"}
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {nid: mod.state_dict() for nid, mod in _model_store["current"].items()}
    torch.save(state, filepath)
    return {"status": "ok", "path": filepath}


def save_model_bytes() -> bytes | None:
    """Serialize trained module state dicts to bytes for download."""
    if "current" not in _model_store:
        return None
    import io
    buf = io.BytesIO()
    state = {nid: mod.state_dict() for nid, mod in _model_store["current"].items()}
    torch.save(state, buf)
    return buf.getvalue()


def load_model_bytes(graph_data: dict, data: bytes) -> dict:
    """Load state dicts from bytes into freshly built modules."""
    import io
    from engine.graph_builder.build import build_modules  # local: avoids import cycle
    saved_states = torch.load(io.BytesIO(data), map_location=get_device(), weights_only=True)
    modules = build_modules(graph_data)
    loaded = 0
    for nid, state_dict in saved_states.items():
        if nid in modules:
            modules[nid].load_state_dict(state_dict)
            loaded += 1
    if loaded == 0:
        return {"error": "No matching layers found — graph structure may have changed"}
    _model_store["current"] = modules
    return {"status": "ok"}


def load_model(graph_data: dict, filepath: str = "storage/weights/current.pt") -> dict:
    """Load saved state dicts into freshly built modules from the graph."""
    if not os.path.exists(filepath):
        return {"error": f"No saved model at {filepath}"}
    saved_states = torch.load(filepath, map_location=get_device(), weights_only=True)
    # Build modules from graph (lightweight — no metadata/results)
    from engine.graph_builder.build import build_modules  # local: avoids import cycle
    modules = build_modules(graph_data)
    # Load saved state dicts
    loaded = 0
    for nid, state_dict in saved_states.items():
        if nid in modules:
            modules[nid].load_state_dict(state_dict)
            loaded += 1
    if loaded == 0:
        return {"error": "No matching layers found — graph structure may have changed"}
    _model_store["current"] = modules
    return {"status": "ok"}
