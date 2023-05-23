#! /usr/bin/env python3

import io
import requests

import torch
import numpy as np

from PIL import Image

from label_studio_ml.model import LabelStudioMLBase

from yogo.model import YOGO


def url_to_img(url: str, normalize_images: bool = False) -> torch.Tensor:
    response = requests.get(url)
    img_bytes = io.BytesIO(response.content)
    img = np.array(Image.open(img_bytes))
    if normalize_images:
        return torch.from_numpy(img[None, None, ...]) / 255
    return torch.from_numpy(img[None, None, ...])


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
        print("A PREDICT CALL", tasks, kwargs)
        predictions = []
        for task in tasks:
            img_url = task["data"]["image"]
            img_ten = url_to_img(img_url, normalize_images=self.normalize_images)

            with torch.no_grad():
                pred = self.model(img_ten.to(self.device))
                print(format_preds(pred))

            predictions.append(
                {
                    "score": 0.987,  # prediction overall score, visible in the data manager columns
                    "model_version": "delorean-20151021",  # all predictions will be differentiated by model version
                    "result": [
                        {
                            "from_name": self.from_name,
                            "to_name": self.to_name,
                            "type": "choices",
                            "score": 0.5,  # per-region score, visible in the editor
                            "value": {"choices": [self.labels[0]]},
                        }
                    ],
                }
            )
        return predictions

    def fit(self, tasks, workdir=None, **kwargs):
        print(tasks, workdir, kwargs)
        return self.pth_path
