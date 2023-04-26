import traceback
import multiprocessing as mp

from tqdm import tqdm
from typing import Any, Sequence, Callable
from functools import partial


def normalize(x, y):
    height = 772
    width = 1032
    return float(x) / width, float(y) / height


def convert_coords(xmin, xmax, ymin, ymax):
    xmin, ymin = normalize(xmin, ymin)
    xmax, ymax = normalize(xmax, ymax)

    if xmin == xmax or ymin == ymax:
        raise ValueError(
            f"xmin == xmax = {xmin == xmax} or ymin == ymax = {ymin == ymax}"
        )

    assert xmin < xmax, f"need xmin < xmax, got {xmin}, {xmax}"
    assert ymin < ymax, f"need ymin < ymax, got {ymin}, {ymax}"

    xcenter = (xmin + xmax) / 2
    ycenter = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    assert all(
        [0 < v < 1 for v in [xcenter, ycenter, width, height]]
    ), f"{[xcenter, ycenter, width, height]}"

    return xcenter, ycenter, width, height


print_lock = mp.Lock()


def protected_fcn(f, *args):
    try:
        f(*args)
    except:
        with print_lock:
            print(f"exception occurred processing {args}")
            print(traceback.format_exc())


def multiprocess_directory_work(
    files: Sequence[Any],
    work_fcn: Callable[
        [
            Any,
        ],
        None,
    ],
):
    fcn = partial(protected_fcn, work_fcn)

    with mp.Pool() as P:
        # iterate so we get tqdm output, thats it!
        vs = []
        for v in tqdm(P.imap_unordered(fcn, files, chunksize=1), total=len(files)):
            vs.append(v)
        return vs
