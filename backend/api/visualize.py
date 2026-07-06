import logging
from fastapi import APIRouter

from engine.graph_builder import get_layer_detail
from visualize.step_through import run_step_through
from visualize.activation_max import activation_maximization
from visualize.backprop_sim import (
    simulate_backprop,
    run_backward_step_through,
    run_one_step,
    run_weight_wiggle,
)
from visualize.loss_landscape import compute_loss_landscape

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["visualize"])


@router.post("/step-through")
async def step_through(request: dict):
    """Run a forward pass through the graph on a single sample, returning ordered stages.

    Request: { graph, filterLabel?, sampleIdx? }
    Response: { status: "ok", result: { stages, sample, sampleIdx } } or error
    """
    logger.info("Step-through requested")
    try:
        result = run_step_through(
            request["graph"],
            filter_label=request.get("filterLabel"),
            sample_idx=request.get("sampleIdx"),
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Step-through failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/backward-step-through")
async def backward_step_through(request: dict):
    """Run a forward+backward pass and return rich per-node backward stages."""
    logger.info("Backward step-through requested")
    try:
        result = run_backward_step_through(
            request["graph"],
            sample_idx=request.get("sampleIdx"),
            filter_label=request.get("filterLabel"),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Backward step-through failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/one-step")
async def one_step(request: dict):
    """Apply one gradient-descent step on a single sample; return before/after.

    Request: { graph, sampleIdx? }
    Response: { status: "ok", result: { lr, loss, prediction, layers, ... } } or error
    """
    logger.info("One-step requested")
    try:
        result = run_one_step(request["graph"], sample_idx=request.get("sampleIdx"))
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"One-step failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/wiggle-weight")
async def wiggle_weight(request: dict):
    """Sweep one weight of a Linear layer; return the loss-vs-weight curve.

    Request: { graph, nodeId, sampleIdx?, weightIndex? }
    Response: { status: "ok", result: { curve, current, step, lr, coord, ... } } or error
    """
    logger.info("Weight wiggle requested")
    try:
        result = run_weight_wiggle(
            request["graph"],
            request["nodeId"],
            sample_idx=request.get("sampleIdx"),
            weight_index=request.get("weightIndex"),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Weight wiggle failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/simulate-backprop")
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


@router.post("/loss-landscape")
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


@router.post("/activation-max")
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


@router.post("/layer-detail")
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
