import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import tifffile as tiff

from pathlib import Path

from yogo.model import YOGO
from yogo.data.utils import read_image


def get_feature_maps(model, image_path, layer_num):
    layers = []
    for i in range(layer_num):
        layers.append(list(model.model.children())[i])

    selected_layers = nn.Sequential(*layers)

    image = read_image(image_path)
    if model.normalize_images:
        image = image / 255
    image = image.unsqueeze(0)

    with torch.no_grad():
        feature_maps = selected_layers(image)

    return feature_maps


def save_feature_maps_to_pdf(feature_maps, output_path, layer_num):
    n, c, h, w = feature_maps.shape

    with PdfPages(output_path) as pdf:
        for i in range(c):
            fig, ax = plt.subplots(figsize=(8, 8))
            cax = ax.imshow(feature_maps[0, i].detach().cpu().numpy(), cmap="gray")
            ax.axis("off")
            plt.title(f"Layer {layer_num} - Feature Map {i+1}")
            fig.colorbar(cax, ax=ax, orientation="vertical")
            pdf.savefig(fig)
            plt.close(fig)


def save_feature_maps_to_tiff(feature_maps, output_path, layer_num):
    _, c, h, w = feature_maps.shape
    feature_maps_np = feature_maps[0].detach().cpu().numpy()
    tiff_path = output_path.parent / f"feature_maps_layer_{layer_num}.tiff"
    tiff.imwrite(tiff_path, feature_maps_np)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("pth_path", type=str)
    parser.add_argument("--output-dir", default=".", type=Path)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--pdf",
        help="save outputs as images in pdfs; useful for just browsing",
        action=argparse.BooleanOptionalAction,
    )
    output_group.add_argument(
        "--tiff",
        help="save outputs as tiff files; each layer's outputs will be one tiff file",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    y, cfg = YOGO.from_pth(
        args.pth_path,
        inference=True,
    )

    num_layers = len(y.model)
    for layer_num in range(1, num_layers + 1):
        feature_maps = get_feature_maps(y, args.image, layer_num)

        output_path = args.output_dir / f"feature_maps_layer_{layer_num}.pdf"

        if args.pdf:
            save_feature_maps_to_pdf(feature_maps, output_path, layer_num)
        elif args.tiff:
            save_feature_maps_to_tiff(feature_maps, output_path, layer_num)
        else:
            raise NotImplementedError("should not happen!")
