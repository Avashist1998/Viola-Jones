# The goal of the file to perform the feature extraction using the Haar_Features. 
# I hoping to define the function in this file and worry about the rest later.
import cv2
import os
import glob 
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
# Setting up a assigning the label to the images
# get the base path of the directory
def intergal_image(image):
    [row,col] = image.shape
    i_image = image.copy()
    for i in range(0,row):
        for j in range(0,col):
            i_image[i,j] = sum(sum(image[0:i+1,0:j+1]))
    return i_image
def feature_extraction(image):
    image_copy = image.copy()
    feature = []
    [row, col] = image_copy.shape
    # Type 1 features (Horizonatal)
    for w in range(1,4):
        for h in range(1,8):
            for i in range(row-h): 
                for j in range(col-2*w):
                    output = -2*image_copy[i+h-1,j+w-1] + 2*image_copy[i,j+w-1] + image_copy[i+h-1,j+2*w-1] + image_copy[i+h-1,j] + image_copy[i,j+2*w-1] - image_copy[i,j]
                    feature.append(output)
    #Type 2 features (Vertical)
    for h in range(1,4):
        for w in range(1,8):
            for i in range(row-2*h): 
                for j in range(col-w):
                    output = 2*image_copy[i+h-1,j] + image_copy[i+2*h-1,j+w-1] + image_copy[i,j+w-1] - 2*image_copy[i+h-1,j+w-1] - image_copy[i+2*h-1,j] - image_copy[i,j]
                    feature.append(output)
    return feature




base_path  =  os.getcwd()
test_faces_list = os.listdir(base_path+ '/dataset/testset/faces')
test_non_faces_list = os.listdir(base_path+ '/dataset/testset/non-faces')
test = cv2.imread( base_path + "/dataset/trainset/faces/face00001.png",cv2.IMREAD_GRAYSCALE)
image = intergal_image(test)
plt.figure
plt.imshow(test)
plt.show()
f = feature_extraction(image)
print('Completed')