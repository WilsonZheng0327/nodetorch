import logging
from fastapi import APIRouter
from viztracer import VizTracer

from engine.graph_builder import execute_graph, infer_graph, evaluate_test_set

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["inference"])


@router.post("/forward")
async def forward(graph_data: dict):
    logger.info("Forward pass requested")
    try:
        # DEV: VizTracer captures the full call tree of this forward pass.
        # Writes forward_trace.json on block exit; view with `vizviewer forward_trace.json`.
        # include_files keeps only our code (skips torch/numpy internals). Remove when done.
        with VizTracer(
            output_file="forward_trace.json",
            include_files=["backend/"],
            max_stack_depth=40,
        ):
            results = execute_graph(graph_data)
        logger.info("Forward pass complete")
        return {"status": "ok", "results": results}
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/infer")
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


@router.post("/evaluate-test")
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
