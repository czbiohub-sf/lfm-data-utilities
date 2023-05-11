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



def set_universal_fig_settings_(fig, scale=0.8):
    fig.update_layout(
        yaxis_visible=False,
        yaxis_showticklabels=False,
        xaxis_visible=False,
        xaxis_showticklabels=False,
        coloraxis_showscale=False,
        width=int(
            (
                objectness_heatmap.shape[1]
                * (image_data.shape[0] / objectness_heatmap.shape[0])
            )
            * scale
        ),
        height=int(772 * scale),
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

    img_fig = px.imshow(bbox_image)
    set_universal_fig_settings_(img_fig)

    app.layout = html.Div([
        html.Div(children="YOGO model output analysis"),
        html.Div(children=f"model: {args.pth_path}"),
        html.Div(children=f"image: {args.image_path}"),
        dcc.Graph(figure=img_fig),
        dcc.Graph(figure={}, id='YOGO-output'),
        dcc.RadioItems(options=['objectness', 'classification', 'bbox'], value='objectness', id='YOGO-output-selector'),
    ])

    @callback(
        Output(component_id='YOGO-output', component_property='figure'),
        Input(component_id='YOGO-output-selector', component_property='value'),
    )
    def update_yogo_output(value):
        # TODO how to get custom hover data
        if value == 'objectness':
            fig = px.imshow(objectness_heatmap)
        elif value == 'classification':
            fig = px.imshow(class_confidences)
            fig.update_traces(showlegend=False)
        elif value == 'bbox':
            raise NotImplementedError

        set_universal_fig_settings_(fig)
        return fig

    app.run_server(debug=True, use_reloader=True)
