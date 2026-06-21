import logging
from fastapi import APIRouter

from persistence.runs_store import list_runs, load_run, delete_run

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["runs"])


@router.get("/runs")
async def get_runs():
    """List all saved training runs (summary only)."""
    return {"status": "ok", "runs": list_runs()}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Load a full training run including per-epoch history."""
    data = load_run(run_id)
    if data is None:
        return {"status": "error", "error": "Run not found"}
    return {"status": "ok", "run": data}


@router.delete("/runs/{run_id}")
async def remove_run(run_id: str):
    ok = delete_run(run_id)
    return {"status": "ok" if ok else "error"}
