#! /usr/bin/env python3


import os
import argparse
import subprocess

from pathlib import Path
from multiprocessing import Process
from http.server import HTTPServer, SimpleHTTPRequestHandler

from labelling_constants import FLEXO_DATA_DIR, IMAGE_SERVER_PORT


try:
    boolean_action = argparse.BooleanOptionalAction  # type: ignore
except AttributeError:
    boolean_action = "store_true"  # type: ignore


def get_parser():
    parser = argparse.ArgumentParser(description="label studio runner!")

    parser.add_argument(
        dest="run_set_folder",
        metavar="run-set-folder",
        nargs="?",
        type=Path,
        help=(
            "path to run set folder (`<some path>/LFM_scope` on flexo), "
            "defaults to running on OnDemand if no argument is provided."
        ),
        default=FLEXO_DATA_DIR,
    )
    return parser


def run_server(directory: Path):
    server_addy = ("localhost", IMAGE_SERVER_PORT)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    httpd = HTTPServer(server_addy, Handler)

    print(
        f"serving your files, Hot n' Fresh, on http://localhost:{IMAGE_SERVER_PORT} "
        f"from {str(directory)}"
    )

    httpd.serve_forever()


def run_server_in_proc(directory: Path) -> Process:
    p = Process(target=run_server, args=(directory,), daemon=True)
    p.start()
    return p


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    path_to_run_folder = args.run_set_folder

    if not path_to_run_folder.exists():
        raise ValueError(
            "warning: your path doesn't exist! Double check that you entered the correct "
            f"path and mounted flexo, got path {path_to_run_folder}"
        )
    elif path_to_run_folder.name == "run-sets":
        print(
            "warning: you provided the path to `run-sets`, not `LFM_scope`. If you're "
            "annotating a LabelStudio tasks file with `run-sets` as the serving dir, "
            "this is fine. Otherwise, if you're loading a new tasks file, you should "
            "restart with the path to `LFM_scope`."
        )
    elif path_to_run_folder.name != "LFM_scope":
        raise ValueError(
            "provided path must be to `flexo/MicroscopyData/Bioengineering/LFM_scope`.\n"
            "When running on OnDemand, this should default to the correct location. Otherwise, make sure you've mounted\n"
            "Flexo, and provide the path to `run-sets`.\n"
            f"got path {path_to_run_folder}"
        )

    os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(path_to_run_folder)

    proc = run_server_in_proc(path_to_run_folder)

    try:
        subprocess.run(["label-studio", "start"])
    except KeyboardInterrupt:
        print("gee wiz, thank you for labelling today!")