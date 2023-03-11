import numpy as np
from json import dump, load
from typing import Dict, List, Callable, Optional

from utils.classifier import Classifier
from utils.decision_stamp import DecisionStamp


class AdaBoost(Classifier):
    """AdaBoost Classifier Definition."""

    def __init__(self, n: int = 5):
        """Initialize the AdaBoost Object."""

        self.num_of_est = n
        self.alphas: List[float] = []
        self.scores: List[float] = []
        self.weighted_errors: List[float] = []
        self.weak_est: List[DecisionStamp] = []

    def _get_adjusted_dist(self, alpha: float, dist: np.array, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Adjusted the dist for the next round."""

        dist = np.multiply(dist, np.exp(alpha * (y != pred)))
        return dist/np.sum(dist)
    
    def __dict__(self) -> Dict[str, str]:
        """Returns a dictionary representation of the object"""

        return {
                "num_of_est": self.num_of_est,
                "alphas": self.alphas,
                "scores": self.scores,
                "weighted_errors": self.weighted_errors,
                "weak_est": [ est.__dict__() for est in self.weak_est ]
                }

    def set_params(self, **kwargs) -> "AdaBoost":
        """Sets the parameters of the classifiers."""

        for parameter, value in kwargs.items():
            setattr(self, parameter, value)
        return self

    def save(self, file_path: str) -> None:
        """Save the AdaBoost model to a file."""

        with open(file_path, "w") as fp:
            dump(self.__dict__(), fp)

    def load(file_path: str) -> "AdaBoost":
        """Loads a AdaBoost Model from file."""

        data = {}
        with open(file_path, "r") as fp:
            data = load(fp)

        for i, weak_est in enumerate(data["weak_est"]):
            data["weak_est"][i] = DecisionStamp().set_params(**weak_est)

        return AdaBoost().set_params(**data)

    def fit(self, X: np.ndarray, y: np.ndarray, dist: np.ndarray = None, /, round_call_back: Optional[Callable[["AdaBoost", int], None]] = None):
        """Train the AdaBoost model on the given input."""

        if dist is None:
            dist = DecisionStamp.init_distribution(y)

        for i in range(self.num_of_est):
            weak_est = DecisionStamp()
            weak_est.fit(X, y, dist)
            pred = weak_est.predict(X)
    
            weak_est_error = weak_est.weighted_error
            alpha = np.log((1-weak_est_error)/(weak_est_error + 1e-18))
            dist = self._get_adjusted_dist(alpha, dist, y, pred)
            
            self.alphas.append(alpha)
            self.weak_est.append(weak_est)
            self.scores.append(self.score(X, y))
            self.weighted_errors.append(weak_est.weighted_error)

            if round_call_back is not None:
                round_call_back(self, i+1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns the accuracy of the model for the given input."""

        pred = self.predict(X)
        equality_check = lambda y1, y2: y1 == y2
        total = sum(map(equality_check, y, pred))
        score = total/len(list(y))
        return score

    def predict(self, X: np.ndarray, /, num_of_est: Optional[int] = None) -> np.ndarray:
        """Classify the data using the specified number of estimators."""

        row, _ = X.shape
        sum_y_pred = np.zeros((row))

        if num_of_est is None:
            num_of_est = len(self.weak_est)
        assert num_of_est <= len(self.weak_est)
        assert num_of_est > 0

        for i, est in enumerate(self.weak_est[:num_of_est]):
            sum_y_pred += self.alphas[i]*est.predict(X)
        y_pred = ((sum_y_pred >= np.sum(self.alphas[:num_of_est])/2)*2 - 1)
        return y_pred
