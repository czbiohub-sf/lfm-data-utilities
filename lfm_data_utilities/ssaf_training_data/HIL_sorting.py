import os
import argparse
import numpy as np
import multiprocessing as mp

import plotly.express as px

from PIL import Image
from tqdm import tqdm
from typing import List
from pathlib import Path
from functools import partial

from dash import DiskcacheManager, Dash, ctx, dcc, html, callback, Input, Output, State

from lfm_data_utilities.ssaf_training_data import utils

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


# Diskcache for non-production apps when developing locally
import diskcache
cache = diskcache.Cache("/tmp/cache")
background_callback_manager = DiskcacheManager(cache)


def set_universal_fig_settings_(fig, img_shape, scale=0.8):
    fig.update_layout(
        yaxis_visible=False,
        yaxis_showticklabels=False,
        xaxis_visible=False,
        xaxis_showticklabels=False,
        coloraxis_showscale=False,
        width=int(img_shape[1] * scale),
        height=int(img_shape[0] * scale),
        margin=dict(l=20, r=20, t=20, b=20),
	)


def process_folder(folder_path: Path, save_loc: Path, focus_graph_loc: Path) -> None:
    """Run the analysis + sorting on a given folder

    Parameters
    ----------
    folder_path: Path
    save_loc: Path
        Where the training data subfolders (i.e displacement folders, [..., +3, +2, ..., -2, -3, ...]) will be saved
    focus_graph_loc: Path
        Where the focus graphs will be saved (optional)
    """

    img_paths = utils.get_list_of_img_paths_in_folder(folder_path)
    motor_positions = utils.get_motor_positions_from_img_paths(img_paths)

    print("Loading images...")
    imgs = utils.load_imgs(img_paths)

    print("Calculating focus metrics...")
    focus_metrics = utils.multiprocess_focus_metric(
        imgs, utils.log_power_spectrum_radial_average_sum
    )

    peak_motor_pos = utils.find_peak_position(
        focus_metrics,
        motor_positions,
        save_loc=focus_graph_loc,
        folder_name=folder_path.stem,
    )

    rel_pos = utils.get_relative_to_peak_positions(motor_positions, peak_motor_pos)
    utils.generate_relative_position_folders(save_loc, rel_pos)

    print("Copying images to their relative position folders...")
    utils.move_imgs_to_relative_pos_folders(img_paths, save_loc, rel_pos)


def multiproc_folders(folders: List[Path], save_loc: Path, focus_graph_loc: Path):
    with mp.Pool() as pool:
        tqdm(
            pool.imap(
                partial(
                    process_folder, save_loc=save_loc, focus_graph_loc=focus_graph_loc
                ),
                folders,
            ),
            total=len(folders),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sort zstacks into training data")
    parser.add_argument(
        "unsorted_zstacks_loc",
        type=Path,
        help="Folder path of zstacks to be sorted",
    )
    parser.add_argument(
        "save_loc",
        type=Path,
        help="Folder path where the training data will be saved (can append to folders in an existing training data directory too)",
    )
    parser.add_argument(
        "focus_graph_loc",
        type=Path,
        help="Folder path where the focus graph plots will be saved",
    )
    args = parser.parse_args()

    if not args.save_loc.exists():
        args.save_loc.mkdir(exist_ok=True, parents=True)

    if not args.focus_graph_loc.exists():
        args.focus_graph_loc.mkdir(exist_ok=True, parents=True)


    folders = utils.get_list_of_zstack_folders(args.unsorted_zstacks_loc)
    num_folders = len(folders)

    # for folder in folders:
    #     try:
    #         process_folder(folder, args.save_loc, args.focus_graph_loc)
    #     except Exception:
    #         import traceback

    #         traceback.print_exc()
    folder_idx = 0
    focus_plot_ids = [f"focus-plot-{i}" for i in range(10)]
    graphs = [
                dcc.Graph(figure={}, id=fpi)
                for fpi in focus_plot_ids
            ]

    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.Div(children="focus picker"),
            html.B( f"folder {folder_idx + 1} / {num_folders}",
                id="folder-progress"
             ),
            html.Div(children=[
                html.Button('back', id='back-btn', n_clicks=0),
                html.Button('next', id='next-btn', n_clicks=0),
            ]),
            html.Div(
                children=[
                    dcc.Graph(figure={}, id="focus-plot"),
                ],
                style={"display": "flex", "flex-direction": "row"},
            ),
            html.Div(children=graphs)
        ]
    )

    @callback(
        Output("focus-plot", "figure"),
        Output("folder-progress", "children"),
        Input("next-btn", "n_clicks"),
        Input("back-btn", "n_clicks"),
    )
    def update_graphs(next_btn, back_btn):
        global folder_idx

        if ctx.triggered_id == "next-btn":
            if folder_idx + 1 >= len(folders):
                return dash.no_update
            folder_idx = folder_idx + 1
        elif ctx.triggered_id == "back-btn":
            if folder_idx == 0:
                return dash.no_update
            folder_idx = folder_idx - 1

        print(f"{ctx.triggered_id} clicked")
        print(f"next_btn: {next_btn}")
        print(f"back_btn: {back_btn}")
        print(f"{folder_idx=}")

        folder_path = folders[folder_idx]
        img_paths = utils.get_list_of_img_paths_in_folder(folder_path)
        motor_positions = utils.get_motor_positions_from_img_paths(img_paths)

        print("Loading images...")
        imgs = utils.load_imgs(img_paths)

        print("Calculating focus metrics...")
        focus_metrics = utils.multiprocess_focus_metric(
            imgs, utils.log_power_spectrum_radial_average_sum
        )

        peak_motor_pos = utils.find_peak_position(
            focus_metrics,
            motor_positions,
            save_loc=args.focus_graph_loc,
            folder_name=folder_path.stem,
        )

        img = Image.open(f"{args.focus_graph_loc / folder_path.stem}.png")
        img_fig = px.imshow(img)
        set_universal_fig_settings_(img_fig, np.array(img).shape, scale=1)

        # rel_pos = utils.get_relative_to_peak_positions(motor_positions, peak_motor_pos)
        # utils.generate_relative_position_folders(save_loc, rel_pos)

        # print("Copying images to their relative position folders...")
        # utils.move_imgs_to_relative_pos_folders(img_paths, save_loc, rel_pos)
        return img_fig,f"folder {folder_idx + 1} / {num_folders}"

    app.run(debug=True, use_reloader=True, port=8053)
