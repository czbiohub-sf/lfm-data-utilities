#! /usr/bin/env python3

import torch

from pathlib import Path

from label_studio_ml.model import LabelStudioMLBase

from yogo.model import YOGO


class YOGOTrainInfer(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pth_path = "/Users/axel.jacobsen/Documents/yogo/trained_models/old-transport-1520/best.pth"
        self.model = YOGO.from_pth(self.pth_path)
        self.model.to(self.device)

        # you can preinitialize variables with keys needed to extract info from tasks and annotations and form predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

    def predict(self, tasks, **kwargs):
        print(tasks, kwargs)
        predictions = []
        for task in tasks:
            predictions.append({
                'score': 0.987,  # prediction overall score, visible in the data manager columns
                'model_version': 'delorean-20151021',  # all predictions will be differentiated by model version
                'result': [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'choices',
                    'score': 0.5,  # per-region score, visible in the editor
                    'value': {
                        'choices': [self.labels[0]]
                    }
                }]
            })
        return predictions

    def fit(self, annotations, **kwargs):
        print(annotations, kwargs)
        return self.pth_path
