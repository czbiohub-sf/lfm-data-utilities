#! /usr/bin/env python3

import os
import io
import requests

import torch
import numpy as np

from PIL import Image

from label_studio_ml.model import LabelStudioMLBase

from yogo.model import YOGO
from yogo.utils.utils import format_preds


def url_to_img(url: str, normalize_images: bool = False) -> torch.Tensor:
    response = requests.get(url)
    img_bytes = io.BytesIO(response.content)
    img = np.array(Image.open(img_bytes))
    if normalize_images:
        return torch.from_numpy(img[None, None, ...]) / 255
    return torch.from_numpy(img[None, None, ...])


# magic to make it work
os.environ["LABEL_STUDIO_ML_BACKEND_V2"] = "true"


class YOGOTrainInfer(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pth_path = "/Users/axel.jacobsen/Documents/yogo/trained_models/old-transport-1520/best.pth"

        self.model, cfg = YOGO.from_pth(self.pth_path)
        self.model.to(self.device)
        self.normalize_images = cfg["normalize_images"]

        # you can preinitialize variables with keys needed to extract info from tasks and annotations and form predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema["to_name"][0]
        self.labels = schema["labels"]

    def predict(self, tasks, **kwargs):
        self.model.inference = True
        predictions = []
        for task in tasks:
            img_url = task["data"]["image"]
            img_ten = url_to_img(img_url, normalize_images=self.normalize_images)

            with torch.no_grad():
                pred = self.model(img_ten.to(self.device))

            results = []
            for i, bbox_pred in enumerate(format_preds(pred[0,...])):
                class_pred = torch.argmax(bbox_pred[5:])
                class_confidence = bbox_pred[class_pred]
                obj_confidence = bbox_pred[4]

                results.append(
                    {
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "original_height": 772,
                        "original_width": 1032,
                        "type": "RectangleLabels",
                        "score": float(class_confidence * obj_confidence),
                        "value": {
                            "x": float(bbox_pred[0]) * 1032,
                            "y": float(bbox_pred[1]) * 772,
                            "width": float(bbox_pred[2]) * 1032,
                            "height": float(bbox_pred[3]) * 772,
                            "rotation": 0,
                            "RectangleLabels": [self.labels[class_pred]]
                        },
                    }
                )

            predictions.append({
                "result": results,
            })

        self.model.inference = False
        return predictions

    def fit(self, *args, **kwargs):
        return {"checkpoint": self.pth_path}
