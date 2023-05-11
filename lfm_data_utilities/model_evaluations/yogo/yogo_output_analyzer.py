#! /usr/bin/env python3

from dash import Dash, dcc, html, callback, Input, Output

import argparse
import numpy as np

from PIL import Image
from pathlib import Path

import plotly.express as px

import yogo


CLASS_LIST = yogo.data.dataset.YOGO_CLASS_ORDERING


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
        margin=dict(l=20, r=20, t=20, b=20),
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
        labels=CLASS_LIST,
    )
    bbox_image = np.array(bbox_image)

    objectness_heatmap = result_tensor[0, 4, :, :].numpy()
    classifications = result_tensor[0, 5:, :, :].numpy()
    max_class_confidences = np.max(classifications, axis=0)

    class_confidence_strings = np.zeros_like(max_class_confidences, dtype=object)
    for j in range(classifications.shape[1]):
        for k in range(classifications.shape[2]):
            class_confidence_strings[j, k] = (
                f"healthy: {classifications[0, j, k]:.2f}<br>"
                f"ring: {classifications[1, j, k]:.2f}<br>"
                f"troph: {classifications[2, j, k]:.2f}<br>"
                f"scizont: {classifications[3, j, k]:.2f}<br>"
                f"gametocyte: {classifications[4, j, k]:.2f}<br>"
                f"wbc: {classifications[5, j, k]:.2f}<br>"
                f"misc: {classifications[6, j, k]:.2f}<br>"
            )
    boxmap = np.array(
        yogo.utils.draw_rects(
            image_data, result_tensor, thresh=0.5, iou_thresh=0, objectness_opacity=1
        )
    )

    app = Dash(__name__)

    img_fig = px.imshow(bbox_image)
    img_fig.update(
        data=[
            {
                "hovertemplate": "<b>%{x}, %{y}</b><extra></extra>",
            }
        ]
    )
    set_universal_fig_settings_(img_fig)

    app.layout = html.Div(
        [
            html.Div(children="YOGO model output analysis"),
            html.Div(children=f"model: {args.pth_path}"),
            html.Div(children=f"image: {args.image_path}"),
            html.Div(
                children=[
                    dcc.Graph(figure=img_fig),
                    dcc.Graph(figure={}, id="YOGO-output"),
                ],
                style={
                    "display": "flex",
                    "flex-direction": "row",
                    "margin": "1rem",
                    "width": "100%",
                },
            ),
            dcc.RadioItems(
                options=["objectness", "classification", "bbox"],
                value="objectness",
                id="YOGO-output-selector",
            ),
            dcc.RadioItems(
                options=["max-confidence", *CLASS_LIST],
                id="class-heatmap-selector",
                value="max-confidence",
            ),
        ]
    )

    @callback(
        Output(component_id="YOGO-output", component_property="figure"),
        Input(component_id="YOGO-output-selector", component_property="value"),
        Input(component_id="class-heatmap-selector", component_property="value"),
    )
    def update_yogo_output(output_value, classification_value):
        if output_value == "objectness":
            fig = px.imshow(objectness_heatmap)
            fig.update(
                data=[
                    {
                        "hovertemplate": "<b>%{z}</b><extra></extra>",
                    }
                ]
            )
        elif output_value == "classification":
            if classification_value == "max-confidence":
                fig = px.imshow(max_class_confidences)
            else:
                class_index = CLASS_LIST.index(classification_value)
                fig = px.imshow(classifications[class_index, :, :])

            fig.update(
                data=[
                    {
                        "customdata": class_confidence_strings,
                        "hovertemplate": "<b>%{customdata}</b><extra></extra>",
                    }
                ]
            )
        elif output_value == "bbox":
            fig = px.imshow(boxmap)
            fig.update_layout(hovermode=False)

        set_universal_fig_settings_(fig)
        return fig

    app.run_server(debug=True, use_reloader=True)
