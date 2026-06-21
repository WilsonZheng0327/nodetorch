import logging
from fastapi import APIRouter

from visualize.latent_viz import generate_latent_grid
from visualize.denoise_viz import run_denoise_step_through
from generate.gan_generate import generate_gan_images
from generate.text_generate import generate_text

logger = logging.getLogger("nodetorch")

router = APIRouter(tags=["generate"])


@router.post("/gan-generate")
async def gan_generate(request: dict):
    """Generate images from a trained GAN."""
    logger.info("GAN generate requested")
    try:
        result = generate_gan_images(
            request["graph"],
            num_samples=request.get("numSamples", 8),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"GAN generate failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/denoise-step-through")
async def denoise_step_through(request: dict):
    """Run DDPM denoising and return images at each timestep."""
    logger.info("Denoise step-through requested")
    try:
        result = run_denoise_step_through(
            request["graph"],
            num_samples=request.get("numSamples", 4),
            capture_every=request.get("captureEvery", 1),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Denoise step-through failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/latent-grid")
async def latent_grid(request: dict):
    """Generate a latent space interpolation grid for a trained VAE."""
    logger.info("Latent grid requested")
    try:
        result = generate_latent_grid(
            request["graph"],
            grid_size=request.get("gridSize", 10),
            latent_range=request.get("latentRange", 3.0),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Latent grid failed: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/generate-text")
async def generate_text_endpoint(request: dict):
    """Generate text autoregressively from a trained language model."""
    logger.info("Text generation requested")
    try:
        result = generate_text(
            request["graph"],
            prompt=request.get("prompt", ""),
            max_tokens=request.get("maxTokens", 200),
            temperature=request.get("temperature", 0.8),
            top_k=request.get("topK", 0),
        )
        if "error" in result:
            return {"status": "error", "error": result["error"]}
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        return {"status": "error", "error": str(e)}
