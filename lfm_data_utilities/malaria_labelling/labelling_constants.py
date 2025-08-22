from pathlib import Path


IMG_WIDTH = 1032
IMG_HEIGHT = 772

IMAGE_SERVER_PORT = 8081

LFM_SCOPE_PATH = Path("/hpc/projects/group.bioengineering/LFM_scope/")
IMG_SERVER_ROOT = LFM_SCOPE_PATH

CLASSES = [
    "healthy",
    "ring",
    "trophozoite",
    "schizont",
    "gametocyte",
    "wbc",
    "misc",
]
