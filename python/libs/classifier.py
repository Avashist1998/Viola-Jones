import numpy as np
from typing import Optional
from abc import ABC, abstractclassmethod


class Classifier(ABC):
    """Abstract class for classification models."""
    @abstractclassmethod
    def __init__(self):
        pass

    @abstractclassmethod
    def fit(self, X: np.ndarray, y: np.ndarray, dist: Optional[np.ndarray]) -> None:
        pass

    @abstractclassmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
