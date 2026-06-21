import logging
from fastapi import APIRouter

from dataprep.tokenizer_preview import preview_tokenizer
from dataprep.data_loaders import DATASET_DETAILS, augmentation_preview

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["data"])


@router.get("/dataset/{dataset_type:path}")
async def dataset_detail(dataset_type: str):
    """Return detailed info about a dataset (labels, sample images, stats)."""
    logger.info(f"Dataset detail requested: {dataset_type}")
    detail_fn = DATASET_DETAILS.get(dataset_type)
    if not detail_fn:
        return {"status": "error", "error": f"Unknown dataset: {dataset_type}"}
    try:
        detail = detail_fn()
        return {"status": "ok", "detail": detail}
    except Exception as e:
        logger.error(f"Dataset detail failed: {e}")
        return {"status": "error", "error": str(e)}


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
