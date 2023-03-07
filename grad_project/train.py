import argparse
import glob 
from typing import Final, List
from os import path, mkdir
import numpy as np
import pandas as pd
import imageio.v2 as imageio

from matplotlib import image

from utils.visualization import draw_feature
from utils.ada_boost import AdaBoost
from utils.logger import init_logger

Logger: Final = init_logger(__name__)

def train_model(dataset_path: str, n: int = 5) -> None:
    """Train the model on a given dataset"""

    clf = AdaBoost(n)
    Logger.info("Loading training dataset")
    train_matrix = load_file(dataset_path)
    Logger.info("Training the model")
    X, y = train_matrix[:, :-1], train_matrix[:, -1]
    clf.fit(X, y)
    return clf


def generating_feature_image(clf: AdaBoost) -> List[np.ndarray]:
    """Generate the images on the features selected."""
    feature_images: List[np.ndarray] = []
    for weak_est in clf.weak_est:
        base_image = imageio.imread("./tests/data/face00001.png")
        feature_images.append(draw_feature(base_image, weak_est.feature_index))
    return feature_images

def load_file(file_path: str) -> np.ndarray:
    return np.loadtxt(file_path, dtype="int", delimiter=",")

def train(dataset_path: str, output_path: str, num_of_est: int) -> None:
    
    if not path.exists(output_path):
        Logger.info("Creating output directory")
        mkdir(output_path)

    MODEL_PATH: Final = path.join(output_path, f"{num_of_est}_round_model.json")
    RESULT_DIR: Final = path.join(output_path, "results")
    IMAGE_DIR: Final = path.join(RESULT_DIR, f"{num_of_est}_round_images")
    TRAIN_PATH: Final = path.join(dataset_path, "train_data.csv")
    
    clf: AdaBoost = train_model(TRAIN_PATH, num_of_est)
    Logger.info("Saving the model")
    clf.save(MODEL_PATH)

    if not path.exists(RESULT_DIR):
        Logger.info("Creating the result dir")
        mkdir(RESULT_DIR)
    
    if not path.exists(IMAGE_DIR):
        Logger.info("Creating the image dir")
        mkdir(IMAGE_DIR)
    
    Logger.info("Saving the feature images")
    features_image = generating_feature_image(clf)
    for i in range(num_of_est):
        image.imsave(f"{IMAGE_DIR}/{i}_round.png", features_image[i])

    Logger.info(f"All images are saved at {IMAGE_DIR}")

def main():
    parser = argparse.ArgumentParser(
        prog = "Dataset creator",
        description= "Creates the training and testing dataset for Face detection from the Yale dataset."
    )
    parser.add_argument("--dataset_path", "-d", type=str)
    parser.add_argument("--output_path", "-o", type=str)
    parser.add_argument("--num_of_est", "-n", type=int, help="number of estimators")
    args = parser.parse_args()
    train(args.dataset_path, args.output_path, args.num_of_est)


if __name__ == "__main__":
    main()
