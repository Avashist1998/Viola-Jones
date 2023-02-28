import argparse
import glob 
from os import path
import numpy as np
import pandas as pd
import imageio.v2 as imageio

from utils.feature_extraction import integral_image, feature_extraction

def create_feature_table(image_folder_path: str) -> pd.DataFrame: 

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
        i_image = integral_image(image)
        f = feature_extraction(i_image)
        face_data = np.append(face_data, f)
    
    num_face_feature = int(len(face_data)/num_face_images)
    face_data = np.resize(face_data, (num_face_images, num_face_feature))

    for non_face_file in non_face_image_files:
        image = imageio.imread(non_face_file)
        i_image = integral_image(image)
        f = feature_extraction(i_image)
        non_face_data = np.append(non_face_data, f)
    
    num_non_face_feature = int(len(non_face_data)/num_non_face_images)
    non_face_data = np.resize(non_face_data, (num_non_face_images, num_non_face_feature))
    
    assert num_face_feature == num_non_face_feature
    
    data = np.concatenate((face_data, non_face_data), axis=0)
    data = np.insert(data, num_non_face_feature, label, axis=1)
    return pd.DataFrame((data).astype(int))


def create_dataset(dataset_path: str, output_path: str) -> None:
    
    print("Starting the creation of test dataset")
    testing_data_path = path.join(dataset_path, "testset")
    testing_csv_filename = path.join(output_path, "test_data.csv")
    print("Extracting test dataset Haar features...")
    test_df = create_feature_table(testing_data_path)
    print("Saving test dataset...")
    test_df.to_csv(testing_csv_filename, header=None, index=None, float_format="%10.5f")
    print(f"Test dataset is save at {testing_csv_filename}")
    
    print("Starting the creation of train dataset")
    training_data_path = path.join(dataset_path, "trainset")
    train_csv_filename = path.join(dataset_path, "train_data.csv")
    print("Extracting train dataset Haar features...")
    train_df = create_feature_table(training_data_path)
    print("Saving train dataset...")
    train_df.to_csv(output_path + "/train_data.csv", header=None, index=None, float_format="%10.5f")
    print(f"Test dataset is save at {train_csv_filename}")


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
