import logging
from fastapi import APIRouter

from export import export_to_python

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["export"])


@router.post("/export-python")
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
