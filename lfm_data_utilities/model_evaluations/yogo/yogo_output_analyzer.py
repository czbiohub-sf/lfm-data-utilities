#! /usr/bin/env python3

import torch

import argparse
import numpy as np

from PIL import Image
from pathlib import Path

import plotly.express as px

import yogo

from dash import Dash, ctx, dcc, html, callback, Input, Output, State


CLASS_LIST = yogo.data.YOGO_CLASS_ORDERING


def set_universal_fig_settings_(fig, img_shape, prediction_shape, scale=0.8):
    fig.update_layout(
        plot_bgcolor="#eeeeee",
        paper_bgcolor="#eeeeee",
        yaxis_visible=False,
        yaxis_showticklabels=False,
        xaxis_visible=False,
        xaxis_showticklabels=False,
        coloraxis_showscale=False,
        width=int((prediction_shape[1] * (img_shape[0] / prediction_shape[0])) * scale),
        height=int(772 * scale),
        margin=dict(l=20, r=20, t=20, b=20),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pth_path", type=Path, help="path to yogo pth file")
    parser.add_argument("image_path", type=Path, help="path to image file")
    parser.add_argument(
        "--port", type=int, default=8050, help="port to run on (default 8050)"
    )
    args = parser.parse_args()

    image_path = args.image_path
    image = Image.open(image_path).convert("L")
    image_data = np.array(image)

    result_tensor = yogo.infer.predict(args.pth_path, path_to_images=image_path,)

    bbox_image = yogo.utils.draw_yogo_prediction(
        torch.tensor(image_data), result_tensor, obj_thresh=0.5, labels=CLASS_LIST,
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
        yogo.utils.draw_yogo_prediction(
            torch.tensor(image_data), result_tensor, obj_thresh=0.5, iou_thresh=0,
        )
    )

    app = Dash(__name__)

    img_fig = px.imshow(bbox_image)
    img_fig.update(data=[{"hovertemplate": "<b>%{x}, %{y}</b><extra></extra>",}])
    set_universal_fig_settings_(img_fig, image_data.shape, objectness_heatmap.shape)

    app.layout = html.Div(
        [
            html.H1(children="YOGO model output analysis"),
            html.Div(
                children=[
                    dcc.Graph(figure=img_fig, id="image-original"),
                    dcc.Graph(figure={}, id="YOGO-output"),
                ],
                style={
                    "display": "flex",
                    "flex-direction": "row",
                    "margin": "1rem",
                    "width": "100%",
                    "background-color": "#eeeeee",
                },
            ),
            html.Div(
                children=[
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
                ],
                style={"display": "flex", "flex-direction": "row",},
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
            fig = px.imshow(objectness_heatmap, zmin=0, zmax=1)
            fig.update(data=[{"hovertemplate": "<b>%{z}</b><extra></extra>",}])
        elif output_value == "classification":
            if classification_value == "max-confidence":
                fig = px.imshow(max_class_confidences, zmin=0, zmax=1)
            else:
                class_index = CLASS_LIST.index(classification_value)
                fig = px.imshow(classifications[class_index, :, :], zmin=0, zmax=1)

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

        set_universal_fig_settings_(fig, image_data.shape, objectness_heatmap.shape)
        return fig

    @callback(
        Output(
            component_id="YOGO-output",
            component_property="figure",
            allow_duplicate=True,
        ),
        Output(component_id="image-original", component_property="figure"),
        Input("image-original", "relayoutData"),
        Input("YOGO-output", "relayoutData"),
        State("YOGO-output", "figure"),
        State("image-original", "figure"),
        prevent_initial_call=True,
    )
    def zoom_event(
        image_original_layout_change, yogo_layout_change, yogo_fig, image_fig
    ):
        img_h, img_w, _ = bbox_image.shape
        try:
            yogo_h, yogo_w = np.array(yogo_fig["data"][0]["z"]).shape
        except KeyError:
            # TODO this is a hack, ideally we could recover the img
            # shape from the b64 encoded img
            yogo_h, yogo_w = img_h, img_w

        if ctx.triggered_id == "image-original":
            relayout_data = image_original_layout_change
            w_ratio = yogo_w / img_w
            h_ratio = yogo_h / img_h
            fig = yogo_fig
        elif ctx.triggered_id == "YOGO-output":
            relayout_data = yogo_layout_change
            w_ratio = img_w / yogo_w
            h_ratio = img_h / yogo_h
            fig = image_fig
        else:
            raise RuntimeError("unknown trigger!")

        try:
            fig["layout"]["xaxis"]["range"] = [
                w_ratio * relayout_data["xaxis.range[0]"],
                w_ratio * relayout_data["xaxis.range[1]"],
            ]
            fig["layout"]["yaxis"]["range"] = [
                h_ratio * relayout_data["yaxis.range[0]"],
                h_ratio * relayout_data["yaxis.range[1]"],
            ]
            fig["layout"]["xaxis"]["autorange"] = False
            fig["layout"]["yaxis"]["autorange"] = False
        except (KeyError, TypeError):
            fig["layout"]["xaxis"]["autorange"] = True
            fig["layout"]["yaxis"]["autorange"] = True

        return yogo_fig, image_fig

    app.run_server(debug=True, use_reloader=True, port=args.port)
