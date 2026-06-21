import logging
from fastapi import APIRouter

from engine.graph_builder import get_device_name, set_device

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/system-info")
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


@router.post("/set-device")
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
