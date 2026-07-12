import logging
from fastapi import APIRouter

from dataprep.tokenizer_preview import preview_tokenizer
from dataprep.data_loaders import augmentation_preview
from dataprep.resolve import resolve_detail

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["data"])


def _detail_response(node: dict):
    """Resolve a data node's detail card and wrap it in the status envelope."""
    try:
        detail = resolve_detail(node)
        if detail is None:
            return {"status": "error", "error": f"Unknown dataset: {node.get('type')}"}
        return {"status": "ok", "detail": detail}
    except Exception as e:
        logger.error(f"Dataset detail failed: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/dataset/{dataset_type:path}")
async def dataset_detail(dataset_type: str):
    """Detail for a built-in dataset by type (no properties). Custom datasets use
    the POST route below, which can carry the node's properties."""
    logger.info(f"Dataset detail requested: {dataset_type}")
    return _detail_response({"type": dataset_type, "properties": {}})


@router.post("/dataset-detail")
async def dataset_detail_post(request: dict):
    """Detail for any data node, including a custom dataset whose behavior depends
    on its properties. Body: { type, id?, properties }."""
    node = {
        "type": request.get("type"),
        "id": request.get("id", ""),
        "properties": request.get("properties") or {},
    }
    logger.info(f"Dataset detail (POST) requested: {node['type']}")
    return _detail_response(node)


@router.post("/tokenizer/preview")
async def tokenizer_preview_endpoint(request: dict):
    """Preview a tokenizer's vocab from the upstream corpus, without training.

    Body: { nodeType, properties, datasetType, sampleText? }
    """
    try:
        node_type = request.get("nodeType")
        properties = request.get("properties") or {}
        dataset_type = request.get("datasetType")
        sample_text = request.get("sampleText")
        if not node_type or not dataset_type:
            return {"status": "error", "error": "nodeType and datasetType required"}
        result = preview_tokenizer(node_type, properties, dataset_type, sample_text)
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Tokenizer preview failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/augmentation-preview")
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
