#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import zarr
from PIL import Image


def main(array_path: str, out_dir: str, n_samples: int):
    array_path = Path(array_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open as a Zarr ARRAY (not group/store)
    z = zarr.open(array_path, mode="r")  # z is a 3D array, you use z[:, :, i]

    if z.ndim != 3:
        raise ValueError(f"Expected 3D array [X, Y, T], got shape {z.shape}")

    X, Y, T = z.shape
    if T == 0:
        raise RuntimeError("Time dimension T is zero; nothing to sample.")

    n_to_sample = min(n_samples, T)
    print(f"Array shape: {z.shape} (X,Y,T)")
    print(f"Sampling {n_to_sample} random time indices out of T={T}")

    rng = np.random.default_rng()
    t_indices = rng.choice(T, size=n_to_sample, replace=False)

    for idx, t in enumerate(t_indices, start=1):
        # Grab the image: z[:, :, t]
        img_2d = z[:, :, t]

        # Ensure 2D
        if img_2d.ndim != 2:
            raise ValueError(f"Slice z[:, :, {t}] is not 2D, got shape {img_2d.shape}")

        # Convert to uint8 for saving
        if not np.issubdtype(img_2d.dtype, np.uint8):
            img_min = float(np.min(img_2d))
            img_max = float(np.max(img_2d))
            if img_max == img_min:
                img_u8 = np.zeros_like(img_2d, dtype=np.uint8)
            else:
                img_norm = (img_2d - img_min) / (img_max - img_min)
                img_u8 = (img_norm * 255).astype(np.uint8)
        else:
            img_u8 = img_2d

        im = Image.fromarray(img_u8)
        out_path = out_dir / f"{t:05d}.png"
        im.save(out_path)

        print(f"[{idx}/{n_to_sample}] Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample random images from a 3D Zarr array [X, Y, T] (z[:, :, i])."
    )
    parser.add_argument(
        "--zip",
        "-z",
        default="zipperzippity",
        help="Path to the Zarr array (default: zipperzippity)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="sub_samples",
        help="Output folder to save sampled PNGs (default: sub_samples)",
    )
    parser.add_argument(
        "--n",
        "-n",
        type=int,
        default=100,
        help="Number of random images to save (default: 100)",
    )

    args = parser.parse_args()
    main(args.array, args.out, args.n)
