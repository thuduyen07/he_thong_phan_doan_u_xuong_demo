from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from backend.inference.contracts import PredictionArtifacts, PredictorSpec


class SegmentationPredictor(ABC):
    def __init__(self, spec: PredictorSpec):
        self.spec = spec

    @abstractmethod
    def predict(self, image: Image.Image) -> PredictionArtifacts:
        raise NotImplementedError
