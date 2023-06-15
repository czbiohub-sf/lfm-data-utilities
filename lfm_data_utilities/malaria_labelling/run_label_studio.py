#! /usr/bin/env python3


import os
import argparse
import warnings
import subprocess

from pathlib import Path
from multiprocessing import Process
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from labelling_constants import FLEXO_DATA_DIR, IMAGE_SERVER_PORT


"""
Label Studio!

There are two main components here:
    1. A simple HTTP server that serves the images in given run set folder
    2. Label Studio itself

This file runs the image server, configures Label Studio, and runs it.
This should only be run on localhost! If you are adapting this for a larger
scale project, you should use a more robust server than the one provided here.
Also, don't accept * for CORS!
"""


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

        def end_headers(self):
            # allowing all access for CORS since we are running on localhost
            # and nobody else should be able to access this server
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    httpd = ThreadingHTTPServer(server_addy, Handler)

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
        warnings.warn(
            "you provided the path to `run-sets`, not `LFM_scope`. If you're "
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
    os.environ["DATA_UPLOAD_MAX_MEMORY_SIZE"] = str(1024 * 1024 * 1024)  # 1GB

    proc = run_server_in_proc(path_to_run_folder)

    try:
        subprocess.run(["label-studio", "start"])
    except KeyboardInterrupt:
        print("gee wiz, thank you for labelling today!")
