# /// script
# requires-python = ">=3.10"
# dependencies = ["zarr<3", "numpy", "pillow"]
# ///

"""Quick tool to save linearly subsampled images from a zarr (useful for cases where a run crashes and the subsample isn't automatically made)."""

import argparse
from pathlib import Path

import numpy as np
import zarr
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zarr_path", type=Path, help="Path to the .zip (zarr) file")
    p.add_argument("-n", "--num", type=int, default=100, help="Number of samples")
    p.add_argument("-o", "--out", type=Path, default=Path("sub_sample_imgs"))
    args = p.parse_args()

    z = zarr.open(str(args.zarr_path), mode="r")
    N = z.initialized
    idxs = np.linspace(0, N - 1, min(args.num, N), dtype=int)

    args.out.mkdir(parents=True, exist_ok=True)
    for i in idxs:
        Image.fromarray(np.asarray(z[..., i])).save(args.out / f"{i:06d}.png")

    print(f"Wrote {len(idxs)} images to {args.out}/")


if __name__ == "__main__":
    main()
