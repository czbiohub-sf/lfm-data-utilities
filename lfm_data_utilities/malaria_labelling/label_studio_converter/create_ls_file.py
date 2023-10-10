import torch

import json
import uuid

from pathlib import Path
from urllib.request import pathname2url
from typing import Any, List, Dict, Tuple, Literal

from yogo.data import YOGO_CLASS_ORDERING

from lfm_data_utilities.utils import path_relative_to

from lfm_data_utilities.malaria_labelling.labelling_constants import (
    IMG_WIDTH,
    IMG_HEIGHT,
    IMAGE_SERVER_PORT,
)
from lfm_data_utilities.malaria_labelling.generate_labelstudio_tasks import (
    LFM_SCOPE_PATH,
)


class LabelStudioTasksFile:
    """
    This file can be used to iteravely create a Label Studio tasks file.

    This is useful for creating tasks files directly from YOGO output.

    Warning: classes must be all indexed the same; sometimes there will be
    labels that are indexed differently
    """

    def __init__(self) -> None:
        self.tasks: List[Dict] = []
        # annotations are human-made labels, predictions are model-made labels
        # so maybe these should be annotations? discuss with team
        self.out_type: Literal["annotations", "predictions"] = "predictions"
        # hard coded because we only label at 772,1032; but we can change this later if needed
        self.IMAGE_WIDTH = IMG_WIDTH
        self.IMAGE_HEIGHT = IMG_HEIGHT

    @staticmethod
    def path_to_url(path: Path) -> str:
        abbreviated_path = str(path_relative_to(Path(path), LFM_SCOPE_PATH))
        return f"http://localhost:{IMAGE_SERVER_PORT}/{pathname2url(abbreviated_path)}"

    def convert_prediction_to_ls(
        self, prediction: Tuple[str, float, float, float, float]
    ) -> Dict[str, Any]:
        """
        converts a prediction to a Label Studio prediction
        prediction should be a list with elements (class_string, xc, yc, w, h)
        class_string is one of YOGO_CLASS_ORDERING
        xc, yc, w, h are floats between 0 and 1
        """
        class_ = prediction[0]
        x = 100 * (prediction[1] - prediction[3] / 2)
        y = 100 * (prediction[2] - prediction[4] / 2)
        w = 100 * prediction[3]
        h = 100 * prediction[4]
        return {
            "id": uuid.uuid4().hex[:10],
            "type": "rectanglelabels",
            "value": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "rotation": 0,
                "rectanglelabels": [class_],
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": self.IMAGE_WIDTH,
            "original_height": self.IMAGE_HEIGHT,
        }

    def add_prediction(
        self,
        image_path: Path,
        predictions: List[Tuple[str, float, float, float, float]],
    ):
        """
        adds a prediction to our running list.

        image_path should be a path to the image, and prediction should be a
        formatted prediction of shape (N, *(xc yc w h *classes)) for that image
        """
        task = {
            "data": {"image": self.path_to_url(image_path)},
            self.out_type: [
                {
                    "result": [
                        self.convert_prediction_to_ls(pred) for pred in predictions
                    ],
                    "ground_truth": False,
                }
            ],
        }
        self.tasks.append(task)

    def write(self, path: Path):
        """
        writes the tasks file to the given path
        """
        if len(self.tasks) == 0:
            raise ValueError("no tasks to write")

        with open(path, "w") as f:
            json.dump(self.tasks, f)


def convert_formatted_YOGO_to_list(
    yogo_preds: torch.Tensor,
) -> List[Tuple[str, float, float, float, float]]:
    "TODO need better name"

    def argmax(ar):
        return max(range(len(ar)), key=ar.__getitem__)

    return [
        (
            YOGO_CLASS_ORDERING[argmax(pred[5:])],
            pred[0].item(),
            pred[1].item(),
            pred[2].item(),
            pred[3].item(),
        )
        for pred in yogo_preds
    ]


if __name__ == "__main__":
    # example usage
    tasks_file = LabelStudioTasksFile()
    tasks_file.add_prediction(
        Path("/hpc/projects/group.bioengineering/LFM_scope/notapath/test_data/1.png"),
        [("ring", 0.5, 0.5, 0.5, 0.5)],
    )
    tasks_file.write(Path("test_tasks.json"))
