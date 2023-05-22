#! /usr/bin/env python3


from label_studio_ml.model import LabelStudioMLBase


class YOGOTrainInfer(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
