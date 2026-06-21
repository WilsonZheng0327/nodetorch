from pathlib import Path

STORAGE_DIR = Path("./storage")
STORAGE_DIR.mkdir(exist_ok=True)
BLOCKS_DIR = STORAGE_DIR / "blocks"
BLOCKS_DIR.mkdir(exist_ok=True)
PRESETS_DIR = STORAGE_DIR / "presets"  # shipped block templates, read-only
