#! /usr/bin/env python3

from dash import Dash, dcc, html, callback, Input, Output

import argparse
import numpy as np

from PIL import Image
from pathlib import Path
from typing import Optional

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

import yogo


def plot_heatmap(yogo_output_array: np.ndarray, z_data: Optional[np.ndarray] = None):
    return px.imshow(
        yogo_output_array,
        # showscale=False,
        # hoverinfo=z_data if z_data is not None else "z",
    )

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
    class_confidences = np.max(classifications, axis=0)

    app = Dash(__name__)

    app.layout = html.Div([
        html.Div(children="YOGO model output analysis"),
        html.Div(children=f"model: {args.pth_path}"),
        html.Div(children=f"image: {args.image_path}"),
        dcc.Graph(figure=px.imshow(bbox_image)),
        dcc.Graph(figure={}, id='YOGO-output'),
        dcc.RadioItems(options=['objectness', 'classification', 'bbox'], value='objectness', id='YOGO-output-selector'),
    ])

    @callback(
        Output(component_id='YOGO-output', component_property='figure'),
        Input(component_id='YOGO-output-selector', component_property='value'),
    )
    def update_yogo_output(value):
        if value == 'objectness':
            return plot_heatmap(objectness_heatmap)
        elif value == 'classification':
            return plot_heatmap(class_confidences)
        elif value == 'bbox':
            raise NotImplementedError

    app.run_server(debug=True, use_reloader=True)
