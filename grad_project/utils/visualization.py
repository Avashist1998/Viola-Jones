import numpy as np
import matplotlib.pyplot as plt

from utils.decision_stamp import DecisionStamp
from utils.feature_extraction import get_feature_values


def draw_feature(base_image: np.ndarray, feature_index: int) -> np.ndarray:
    """Draw the feature on a given image."""
    row, col = base_image.shape
    x, y, w, h, feature_type = get_feature_values(row, col, feature_index)
    if feature_type == "Vertical edge feature":
        base_image[x:x+h, y:y+w] = 255
        base_image[x:x+h, y+w:y+2*w] = 0

    elif feature_type == "Horizontal edge feature":
        base_image[x:x+h, y:y+w] = 255
        base_image[x+h:x+2*h, y:y+w] = 0

    elif feature_type == "Vertical band feature":
        base_image[x:x+h, y:y+w] = 255
        base_image[x:x+h, y+w:y+3*w] = 0
        base_image[x:x+h, y+3*w:y+4*w] = 255

    elif feature_type == "Horizontal band feature":
        base_image[x:x+h, y:y+w] = 255
        base_image[x+h:x+3*h, y:y+w] = 0
        base_image[x+3*h:x+4*w, y:y+w] = 255
    else:
        base_image[x:x+h, y:y+w] = 255
        base_image[x+h:x+2*h, y:y+w] = 0
        base_image[x:x+h, y+w:y+2*w] = 0
        base_image[x+h:x+2*h, y+w:y+2*w] = 255

    return base_image


def decision_stamp_visualization(X: np.ndarray, y: np.ndarray, dist: np.ndarray, est: DecisionStamp, /, title_text: str = "Decision Stamp"):
    """Decision Stamp Visualization"""

    fig = plt.figure()
    prediction = est.predict(X)
    color = np.array(["r"]*len(y))
    color[np.where(prediction == y)] = "b"
    plt.scatter(X[:, est.feature_index], y, s=np.sqrt(dist)*1500, color=color)
    plt.axvline(x=est.theta, color="g", label="theta")
    plt.title(title_text)
    plt.xlabel(f"feature_index : {est.feature_index}")
    plt.ylabel("classification")
    plt.show()


def draw_square(base_image: np.ndarray, i: int, y: int, w: int, h: int, line_thick: int) -> np.ndarray:
    """Draws a square on the classification."""

    dim = base_image.shape
    return_image = base_image.copy()

    if len(dim) == 3:
        return_image[i:i+line_thick, y:y+w, 0] = 255
        return_image[i:i+line_thick, y:y+w, 1:] = 0
        return_image[i+h-line_thick:i+h, y:y+w, 0] = 255
        return_image[i+h-line_thick:i+h, y:y+w, 1:] = 0
        return_image[i:i+h, y:y+line_thick, 0] = 255
        return_image[i:i+h, y:y+line_thick, 1:] = 0
        return_image[i:i+h, y+w-line_thick:y+w, 0] = 255
        return_image[i:i+h, y+w-line_thick:y+w, 1:] = 0
    else:
        red_max = int(255*0.299)
        return_image[i:i+line_thick, y:y+w] = red_max
        return_image[i+h-line_thick:i+h, y:y+w] = red_max
        return_image[i:i+h, y:y+line_thick] = red_max
        return_image[i:i+h, y+w-line_thick:y+w] = red_max

    return return_image
