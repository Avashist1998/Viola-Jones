# The goal of the file to perform the feature extraction using the Haar_Features. 
# I hoping to define the function in this file and worry about the rest later.
import cv2
import os
import glob 
from PIL import Image
import numpy as np 
import matplotlib 
import pandas as pd 
# Setting up a assigning the label to the images
# get the base path of the directory
def intergal_image(image):
    [row,col,dem] = image.shape
    i_image = image
    if (dem == 1):
        for i in range(0,row):
            for j in range(0,col):
                i_image[i,j,0] = sum(sum(image[0:i+1,0:j+1,0]))
    else:
        for i in range(0,row):
            for j in range(0,col):
                i_image[i,j,0] = sum(sum(image[0:i+1,0:j+1,0]))
                i_image[i,j,1] = sum(sum(image[0:i+1,0:j+1,1]))
                i_image[i,j,2] = sum(sum(image[0:i+1,0:j+1,2]))
    return i_image

base_path  =  os.getcwd()
test_faces_list = os.listdir(base_path+ '/dataset/testset/faces')
test_non_faces_list = os.listdir(base_path+ '/dataset/testset/non-faces')
test = cv2.imread( base_path + "/dataset/face1.jpg",cv2.IMREAD_COLOR)
image = intergal_image(test)
cv2.imshow('image',image)

def Haan_creater(size,type):
    print('works')