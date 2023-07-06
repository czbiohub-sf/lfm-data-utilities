from pathlib import Path


IMG_WIDTH = 1032
IMG_HEIGHT = 772

IMAGE_SERVER_PORT = 8081

FLEXO_DATA_DIR = Path("/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/")
IMG_SERVER_ROOT = FLEXO_DATA_DIR

CLASSES = [
    "healthy",
    "ring",
    "trophozoite",
    "schizont",
    "gametocyte",
    "wbc",
    "misc",
]
