#! /usr/bin/env python3

import argparse
import numpy as np

from PIL import Image
from pathlib import Path

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

import yogo


def plot_heatmap(image_data, objectness_data, class_data, scale=0.8):
    # Calculate the total aspect ratio and the proportion for each subplot
    image_aspect = image_data.shape[1] / image_data.shape[0]
    heatmap_aspect = objectness_data.shape[1] / objectness_data.shape[0]

    total_aspect = image_aspect + heatmap_aspect
    image_proportion = image_aspect / total_aspect

    # Create subplots with 1 row and 2 columns
    spacing = 0.04
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["image", "objectness", "", "class predictions"],
        specs=[[{"type": "image"}, {"type": "heatmap"}], [None, {"type": "heatmap"}]],
        horizontal_spacing=spacing,
        vertical_spacing=spacing,
    )

    fig.add_trace(px.imshow(image_data).data[0], row=1, col=1)

    fig.add_trace(
        go.Heatmap(
            z=objectness_data,
            showscale=False,
            colorscale="Viridis",
            hoverinfo="z",
            xaxis="x2",
            yaxis="y2",
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=class_data,
            showscale=False,
            colorscale="Viridis",
            hoverinfo="z",
            xaxis="x3",
            yaxis="y3",
        ),
        row=1,
        col=2,
    )

    # Update layouts so the heatmaps ahve same heights, maintain aspect ratios,
    # and in general make the plot pretty
    fig.update_layout(
        yaxis_visible=False,
        yaxis_showticklabels=False,
        xaxis_visible=False,
        xaxis_showticklabels=False,
        yaxis2_visible=False,
        yaxis2_showticklabels=False,
        xaxis2_visible=False,
        xaxis2_showticklabels=False,
        yaxis3_visible=False,
        yaxis3_showticklabels=False,
        xaxis3_visible=False,
        xaxis3_showticklabels=False,
        coloraxis_showscale=False,
        xaxis=dict(domain=[0, image_proportion]),
        xaxis2=dict(domain=[image_proportion + spacing, 1]),
        xaxis3=dict(domain=[image_proportion + spacing, 1]),
        yaxis=dict(domain=[0, 0.5]),
        yaxis2=dict(domain=[0, 0.5]),
        yaxis3=dict(domain=[0.5 + spacing, 1]),
        width=int(
            (
                1032
                + objectness_data.shape[1]
                * (image_data.shape[0] / objectness_data.shape[0])
            )
            * scale
        ),
        height=int(772 * 2 * scale),
    )

    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pth_path", type=Path, help="path to yogo pth file")
    parser.add_argument("image_path", type=Path, help="path to image file")
    args = parser.parse_args()

    image_path = args.image_path
    image = Image.open(image_path).convert("L")
    image_data = np.array(image)

    result_tensor = yogo.infer.predict(
        args.pth_path,
        path_to_images=image_path,
    )

    bbox_image = yogo.utils.draw_rects(
        image_data,
        result_tensor,
        thresh=0.5,
        labels=yogo.data.dataset.YOGO_CLASS_ORDERING,
    )
    bbox_image = np.array(bbox_image)

    objectness_heatmap = np.flipud(result_tensor[0, 4, :, :].numpy())
    classifications = result_tensor[0, 5:, :, :].numpy()
    class_confidences = np.flipud(np.max(classifications, axis=0))

    plot_heatmap(bbox_image, objectness_heatmap, class_confidences)
