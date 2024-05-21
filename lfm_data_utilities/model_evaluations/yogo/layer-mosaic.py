import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from yogo.model import YOGO
from yogo.data.utils import read_image
from matplotlib.backends.backend_pdf import PdfPages


def save_feature_maps_to_pdf(feature_maps, output_path, layer_num):
    n, c, h, w = feature_maps.shape

    with PdfPages(output_path) as pdf:
        for i in range(c):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(feature_maps[0, i].detach().cpu().numpy(), cmap="gray")
            ax.axis("off")
            plt.title(f"Layer {layer_num} - Feature Map {i+1}")
            pdf.savefig(fig)
            plt.close(fig)


def save_layer_feature_maps_to_pdf(model, image_path, output_path, layer_num):
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

    save_feature_maps_to_pdf(feature_maps, output_path, layer_num)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("pth_path", type=str)
    parser.add_argument("--output-dir", default=".", type=Path)
    args = parser.parse_args()

    y, cfg = YOGO.from_pth(
        args.pth_path,
        inference=True,
    )

    num_layers = len(y.model)
    for layer_num in range(1, num_layers + 1):
        output_path = args.output_dir / f"feature_maps_layer_{layer_num}.pdf"
        save_layer_feature_maps_to_pdf(y, args.image, output_path, layer_num)
