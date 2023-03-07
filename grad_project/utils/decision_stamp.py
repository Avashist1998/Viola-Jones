import numpy as np
from json import dump
from typing import Optional, Tuple, Dict
from utils.classifier import Classifier

class DecisionStamp(Classifier):

    def __init__(self):
        self.dist = None
        self.polarity = 1
        self.theta = float("inf")
        self.feature_index = None
        self.weighted_error = float("inf")

    def __dict__(self) -> Dict[str, str]:
        return {"feature_index": self.feature_index,
                "theta": self.theta,
                "polarity": self.polarity,
                "weighted_error": self.weighted_error}

    def set_params(self, **kwargs) -> "DecisionStamp":
        """Sets the parameters of the classifiers."""
        for parameter, value in kwargs.items():
            setattr(self, parameter, value)
        return self

    def save(self, file_path: str) -> None:
        """Save the decision tree model."""
        with open(file_path, "w") as fp:
            dump(self.__dict__(), fp)


    def init_distribution(labels: np.ndarray) -> np.ndarray:
        classes, counts = np.unique(labels, return_counts=True)
        count_map = dict(zip(classes, counts))
        init_distribution = np.array([1/(2*count_map[1])]*count_map[1] + [1/(2*count_map[-1])]*count_map[-1])
        return init_distribution

    def fit(self, X: np.ndarray, y: np.ndarray, dist: Optional[np.ndarray] = None):
        if dist is None:
            dist = DecisionStamp.init_distribution(y)
        self.dist = dist
        self.weighted_error, self.feature_index, self.theta = self._find_theta_and_f_star(X, y, dist)

    def predict(self, X:np.ndarray) -> np.ndarray:
        return np.array([ 1 if x <= self.theta else -1 for x in X[:, self.feature_index]])*self.polarity

    def score(self, label, prediction):
        same_check = lambda y1, y2: y1 == y2
        total = sum(map(same_check, label, prediction))
        score = total/len(list(label))
        return score

    def _find_theta_and_f_star(self, X: np.ndarray, y: np.ndarray, dist:np.ndarray) -> Tuple[float, float]:
        row, col = X.shape
        labels, counts = np.unique(y, return_counts=True)
        count_map = dict(zip(labels, counts))
        if (count_map[1] > count_map[-1]):
            y = -1*y
            self.polarity = -1
        F_star = float('inf')
        for j in range(0, col):
            sort_order = X[:,j].argsort()
            Xj =  (X[:,j])[sort_order]
            Yj =  (y)[sort_order]
            Dj = (dist)[sort_order]
            F = sum(Dj[Yj == 1])
            if F < F_star:
                F_star = F
                theta_star = Xj[0]-1
                j_star = j
            for i in range(0, row-1):
                F = F - Yj[i]*Dj[i]
                if ((F < F_star) &  (Xj[i] != Xj[i+1])):
                    F_star = F
                    theta_star = 0.5*((Xj[i] + Xj[i+1]))
                    j_star = j
        return(F_star, j_star, theta_star)