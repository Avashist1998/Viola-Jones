import argparse
import glob 
from os import path, mkdir
import numpy as np
import pandas as pd
from typing import Final 
import imageio.v2 as imageio

from utils.feature_extraction import extract_image_features
from utils.logger import init_logger


Logger: Final = init_logger(__name__)

def create_feature_df(image_folder_path: str) -> pd.DataFrame:

    face_data, non_face_data = np.array([[]]), np.array([[]])
    face_image_files = glob.glob(image_folder_path + "/faces/*.png")
    non_face_image_files = glob.glob(image_folder_path + "/non-faces/*.png")
    face_image_files.sort()
    non_face_image_files.sort()
    
    num_face_images = len(face_image_files)
    num_non_face_images = len(non_face_image_files)
    label = np.append([1]*num_face_images, [-1]*num_non_face_images)

    for face_file in face_image_files:
        image = imageio.imread(face_file)
        f = extract_image_features(image)
        face_data = np.append(face_data, f)
    
    num_face_feature = int(len(face_data)/num_face_images)
    face_data = np.resize(face_data, (num_face_images, num_face_feature))

    for non_face_file in non_face_image_files:
        image = imageio.imread(non_face_file)
        f = extract_image_features(image)
        non_face_data = np.append(non_face_data, f)
    
    num_non_face_feature = int(len(non_face_data)/num_non_face_images)
    non_face_data = np.resize(non_face_data, (num_non_face_images, num_non_face_feature))
    
    assert num_face_feature == num_non_face_feature
    
    data = np.concatenate((face_data, non_face_data), axis=0)
    data = np.insert(data, num_non_face_feature, label, axis=1)
    return pd.DataFrame((data).astype(int))


def create_dataset(dataset_path: str, output_path: str) -> None:

    if not path.exists(output_path):
        Logger.info("Creating Output directory")
        mkdir(output_path)
    Logger.info("Starting the creation of test dataset")
    testing_data_path = path.join(dataset_path, "testset")
    testing_csv_filename = path.join(output_path, "test_data.csv")
    Logger.info("Extracting test dataset Haar features...")
    test_df = create_feature_df(testing_data_path)
    Logger.info("Saving test dataset...")
    test_df.to_csv(testing_csv_filename, header=None, index=None, float_format="%10.5f")
    Logger.info(f"Test dataset is saved at {testing_csv_filename}")
    
    Logger.info("Starting the creation of train dataset")
    training_data_path = path.join(dataset_path, "trainset")
    train_csv_filename = path.join(output_path, "train_data.csv")
    Logger.info("Extracting train dataset Haar features...")
    train_df = create_feature_df(training_data_path)
    Logger.info("Saving train dataset...")
    train_df.to_csv(train_csv_filename, header=None, index=None, float_format="%10.5f")
    Logger.info(f"Train dataset is saved at {train_csv_filename}")


def main():
    parser = argparse.ArgumentParser(
        prog = "Dataset creator",
        description= "Creates the training and testing dataset for Face detection from the Yale dataset."
    )
    parser.add_argument("--dataset_path", "-d", type=str)
    parser.add_argument("--output_path", "-o", type=str)
    args = parser.parse_args()
    create_dataset(args.dataset_path, args.output_path)


if __name__ == "__main__":
    main()
