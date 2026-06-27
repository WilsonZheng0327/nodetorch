import logging
from fastapi import APIRouter

from engine.graph_builder import infer_graph, evaluate_test_set

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["inference"])


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
