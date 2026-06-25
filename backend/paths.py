import os
from pathlib import Path

# Anchor storage to the REPO ROOT (this file lives in backend/), not the current
# working directory. A CWD-relative path created two separate storage trees
# depending on whether you launched from backend/ or the repo root — so datasets,
# weights, and runs were silently duplicated. Override with NODETORCH_STORAGE.
_REPO_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.environ.get("NODETORCH_STORAGE", _REPO_ROOT / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_DIR = STORAGE_DIR / "datasets"          # downloaded datasets (MNIST, CIFAR, …)
DATASETS_DIR.mkdir(exist_ok=True)
BLOCKS_DIR = STORAGE_DIR / "blocks"              # user-saved custom blocks
BLOCKS_DIR.mkdir(exist_ok=True)
BLOCK_TEMPLATES_DIR = STORAGE_DIR / "block-templates"  # shipped block templates (committed)
