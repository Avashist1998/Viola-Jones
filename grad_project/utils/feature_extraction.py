import numpy as np
from typing import List, Tuple, Optional


def get_integral_image(image: np.ndarray) -> np.ndarray:
    """Computes the integral image of an image."""

    [row, col] = image.shape
    i_image: np.ndarray = image.copy().astype("int32")
    for j in range(1, col):
        i_image[:, j] = np.int32(i_image[:, j-1] + i_image[:, j])

    for i in range(1, row):
        i_image[i, :] = np.int32(i_image[i-1, :] + i_image[i, :])

    return i_image


def get_sum_from_i_image(x: int, y: int, w: int, h: int, i_image: np.ndarray) -> np.int32:
    """Returns the sum of an area from the given i image"""

    i_image_padded = np.pad(i_image, ((1, 0), (1, 0)), constant_values=(0))
    A = i_image_padded[x, y]
    B = i_image_padded[x, y+w]
    C = i_image_padded[x+h, y]
    D = i_image_padded[x+h, y+w]
    return D + A - (B + C)


def get_v_edge_feature(x: int, y: int, w: int, h: int, i_image: np.ndarray) -> np.int32:
    """Calculates the vertical edge features."""

    return get_sum_from_i_image(x, y, w, h, i_image) - get_sum_from_i_image(x, y+w, w, h, i_image)


def get_h_edge_feature(x: int, y: int, w: int, h: int, i_image: np.ndarray) -> np.int32:
    """Calculates the horizontal edge features."""

    return get_sum_from_i_image(x, y, w, h, i_image) - get_sum_from_i_image(x+h, y, w, h, i_image)


def get_v_band_feature(x: int, y: int, w: int, h: int, i_image: np.ndarray) -> np.int32:
    """Calculates the vertical band features."""

    white_area = (get_sum_from_i_image(x, y, w, h, i_image) +
                  get_sum_from_i_image(x, y+2*w, w, h, i_image))
    black_area = get_sum_from_i_image(x, y+w, w, h, i_image)
    return white_area - black_area


def get_h_band_feature(x: int, y: int, w: int, h: int, i_image: np.ndarray) -> np.int32:
    """Calculates the horizontal band features."""

    white_area = (get_sum_from_i_image(x, y, w, h, i_image) +
                  get_sum_from_i_image(x+2*h, y, w, h, i_image))
    black_area = get_sum_from_i_image(x+h, y, w, h, i_image)
    return white_area - black_area


def get_slant_edge_feature(x: int, y: int, w: int, h: int, i_image: np.ndarray) -> np.int32:
    """Calculates the slant edge features."""

    white_area = (get_sum_from_i_image(x, y, w, h, i_image) +
                  get_sum_from_i_image(x+h, y+w, w, h, i_image))
    black_area = (get_sum_from_i_image(x+h, y, w, h, i_image) +
                  get_sum_from_i_image(x, y+w, w, h, i_image))
    return white_area - black_area


def extract_i_image_features(i_image: np.ndarray) -> List[int]:
    """Extract haar features from integral image."""

    row, col = i_image.shape
    feature_val: List[int] = []
    max_height, max_width = row+1, col//2+1
    for feature_width in range(1, max_width):
        for feature_height in range(1, max_height):
            height_limit, width_limit = row-1*feature_height+1, col-2*feature_width+1
            for i in range(height_limit):
                for j in range(width_limit):
                    feature_val.append(get_v_edge_feature(
                        i, j, feature_width, feature_height, i_image))

    max_height, max_width = row//2+1, col+1
    for feature_width in range(1, max_width):
        for feature_height in range(1, max_height):
            height_limit, width_limit = row - 2*feature_height+1, col - 1*feature_width+1
            for i in range(height_limit):
                for j in range(width_limit):
                    feature_val.append(get_h_edge_feature(
                        i, j, feature_width, feature_height, i_image))

    max_height, max_width = row+1, col//3+1
    for feature_width in range(1, max_width):
        for feature_height in range(1, max_height):
            height_limit, width_limit = row-1*feature_height+1, col-3*feature_width+1
            for i in range(height_limit):
                for j in range(width_limit):
                    feature_val.append(get_v_band_feature(
                        i, j, feature_width, feature_height, i_image))

    max_height, max_width = row//3+1, col+1
    for feature_width in range(1, max_width):
        for feature_height in range(1, max_height):
            height_limit, width_limit = row-3*feature_height+1, col - 1*feature_width+1
            for i in range(height_limit):
                for j in range(width_limit):
                    feature_val.append(get_h_band_feature(
                        i, j, feature_width, feature_height, i_image))

    max_height, max_width = row//2+1, col//2+1
    for feature_width in range(1, max_width):
        for feature_height in range(1, max_height):
            height_limit, width_limit = row-2*feature_height+1, col-2*feature_width+1
            for i in range(height_limit):
                for j in range(width_limit):
                    feature_val.append(get_slant_edge_feature(
                        i, j, feature_width, feature_height, i_image))


def extract_image_features(image: np.ndarray) -> List[int]:
    """Extract features from a given image"""

    i_image = get_integral_image(image)
    feature_val = extract_i_image_features(i_image)
    return feature_val


def get_feature_values(height: int, width: int, feature_index: int) -> Optional[Tuple[int, int, int, int, str]]:
    val = 0
    max_height, max_width = height+1, width//2+1
    for feature_width in range(1, max_width):
        for feature_height in range(1, max_height):
            height_limit, width_limit = height-1*feature_height+1, width-2*feature_width+1
            for i in range(0, height_limit):
                for j in range(0, width_limit):
                    if val == feature_index:
                        return (i, j, feature_width, feature_height, "Vertical edge feature")
                    val += 1
    max_height, max_width = height//2+1, width+1
    for feature_width in range(1, max_width):
        for feature_height in range(1, max_height):
            height_limit, width_limit = height - 2 * \
                feature_height+1, width - 1*feature_width+1
            for i in range(height_limit):
                for j in range(width_limit):
                    if val == feature_index:
                        return (i, j, feature_width, feature_height, "Horizontal edge feature")
                    val += 1

    max_height, max_width = height+1, width//3+1
    for feature_width in range(1, max_width):
        for feature_height in range(1, max_height):
            height_limit, width_limit = height-1*feature_height+1, width - 3*feature_width+1
            for i in range(height_limit):
                for j in range(width_limit):
                    if val == feature_index:
                        return (i, j, feature_width, feature_height, "Vertical band feature")
                    val += 1

    max_height, max_width = height//3+1, width+1
    for feature_width in range(1, max_width):
        for feature_height in range(1, max_height):
            height_limit, width_limit = height-3*feature_height+1, width - 1*feature_width+1
            for i in range(height_limit):
                for j in range(width_limit):
                    if val == feature_index:
                        return (i, j, feature_width, feature_height, "Horizontal band feature")
                    val += 1

    max_height, max_width = height//2+1, width//2+1
    for feature_width in range(1, max_width):
        for feature_height in range(1, max_height):
            height_limit, width_limit = height-2*feature_height+1, width-2*feature_width+1
            for i in range(height_limit):
                for j in range(width_limit):
                    if val == feature_index:
                        return (i, j, feature_width, feature_height, "Slant feature")
                    val += 1