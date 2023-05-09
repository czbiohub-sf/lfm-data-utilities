#! /usr/bin/env python3

import argparse
import plotly.subplots as sp
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    # Load your grayscale image
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path, help="path to image file")
    args = parser.parse_args()

    image_path = args.image_path
    image = Image.open(image_path).convert("L")
    image_data = np.array(image)

    # Replace these with your YOLO-like object detector's output data
    heatmap_data = np.random.rand(97, 129)  # Example data

    # Calculate the total aspect ratio and the proportion for each subplot
    image_aspect = image_data.shape[1] / image_data.shape[0]
    heatmap_aspect = heatmap_data.shape[1] / heatmap_data.shape[0]

    total_aspect = image_aspect + heatmap_aspect
    image_proportion = image_aspect / total_aspect
    heatmap_proportion = heatmap_aspect / total_aspect

    # Create subplots with 1 row and 2 columns
    fig = sp.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Grayscale Image", "Heatmap"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
        horizontal_spacing=0.02,  # Adjust this as needed
    )

    # Add grayscale image to the left
    fig.add_trace(
        go.Heatmap(
            z=image_data,
            colorscale="gray",
            showscale=False,
            xaxis="x1",
            yaxis="y1",
        ),
        row=1,
        col=1,
    )

    # Add heatmap to the right
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            colorscale="Viridis",
            hoverinfo="z",
            xaxis="x2",
            yaxis="y2",
        ),
        row=1,
        col=2,
    )

    # Set domain based on the calculated proportions
    fig.update_layout(
        xaxis=dict(domain=[0, image_proportion]),
        xaxis2=dict(domain=[image_proportion + 0.02, 1]),  # 0.02 is the spacing
        yaxis=dict(domain=[0, 1]),
        yaxis2=dict(domain=[0, 1]),
        width=int(1032 * 2 * 3/4),
        height=int(772 * 3/4),
    )

    fig.show()
