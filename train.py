import os
from utils.AdaBoost import ada_boost
from utils.utils import intergal_image, feature_extraction





base_path  =  os.getcwd()
train_faces_files = glob.glob(base_path+ '/dataset/trainset/faces/*.png')
train_faces_files.sort()
train_non_faces_files = glob.glob(base_path+ '/dataset/trainset/non-faces/*.png')
train_non_faces_files.sort()
