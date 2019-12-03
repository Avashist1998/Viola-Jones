# The goal of the file to perform the feature extraction using the Haar_Features. 
# I hoping to define the function in this file and worry about the rest later.
import cv2
import os
import glob 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import time
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
    # Type 1 features (two vertical)
    for w in range(1,5):
        for h in range(1,9):
            for i in range(row-h+1): 
                for j in range(col-2*w+1):
                    output = -2*image_copy[i+h-1,j+w-1] + 2*image_copy[i,j+w-1] + image_copy[i+h-1,j+2*w-1] + image_copy[i+h-1,j] + image_copy[i,j+2*w-1] - image_copy[i,j]
                    feature.append(output)
    #Type 2 features (two horizontal)
    for h in range(1,5):
        for w in range(1,9):
            for i in range(row-2*h+1): 
                for j in range(col-w+1):
                    output = 2*image_copy[i+h-1,j] + image_copy[i+2*h-1,j+w-1] + image_copy[i,j+w-1] - 2*image_copy[i+h-1,j+w-1] - image_copy[i+2*h-1,j] - image_copy[i,j]
                    feature.append(output)
    # Type 3 feature (three Horizonatal)
    for h in range(1,3):
        for w in range(1,9):
            for i in range(row-4*h+1): 
                for j in range(col-w+1):
                    output = 2*image_copy[i+3*h-1,j+w-1] + 2*image_copy[i+h-1,j] - 2*image_copy[i+h-1,j+w-1] - 2*image_copy[i+3*h-1,j] - image_copy[i+4*h-1,j+w-1] + image_copy[i+4*h-1,j] - image_copy[i,j] + image_copy[i,j+w-1]
                    feature.append(output)
    # Type 4 feature (two Vertical)
    for h in range(1,9):
        for w in range(1,3):
            for i in range(row-h+1): 
                for j in range(col-4*w+1):
                    output = 2*image_copy[i,j+w-1] + 2*image_copy[i+h-1,j+3*w-1] - 2*image_copy[i,j+3*w-1] - 2*image_copy[i+h-1,j+w-1] - image_copy[i,j]+ image_copy[i+h-1,j] - image_copy[i+h-1,j+4*w-1] + image_copy[i,j+4*w-1]
                    feature.append(output)
    # Type 5 feature (four)
    for h in range(1,5):
        for w in range(1,5):
            for i in range(row-2*h+1): 
                for j in range(col-2*w+1):
                    output = image_copy[i,j]+ 4*image_copy[i+h-1,j+w-1] - 2*image_copy[i,j+w-1] - 2*image_copy[i+h-1,j] + image[i+2*h-1,j+2*w-1] - 2*image_copy[i+h-1,j+2*w-1] + image_copy[i,j+2*w-1] - 2*image_copy[i+2*h-1,j+w-1] + image_copy[i+2*h-1,j]
                    feature.append(output)
    
    return feature


base_path  =  os.getcwd()
train_faces_files = glob.glob(base_path+ '/dataset/trainset/faces/*.png')
train_faces_files.sort()
train_non_faces_files = glob.glob(base_path+ '/dataset/trainset/non-faces/*.png')
train_non_faces_files.sort()
data = np.array([[]])
t0 = time.time()
for names in train_faces_files:
    image = cv2.imread(names,cv2.IMREAD_GRAYSCALE)
    i_image = intergal_image(image)
    f = feature_extraction(i_image)
    data = np.append(data,f)

num_image = len(train_faces_files)
num_feature = int(len(data)/num_image)
data = np.resize(data, (num_image,num_feature))
temp_data = np.array([[]])
for names in train_non_faces_files:
    image = cv2.imread(names,cv2.IMREAD_GRAYSCALE)
    i_image = intergal_image(image)
    f = feature_extraction(i_image)
    temp_data = np.append(temp_data,f)
num_image = int(len(temp_data)/num_feature)
temp_data = np.resize(temp_data, (num_image,num_feature))
label = [1]*len(train_faces_files)
label_non_faces = [-1] * len(train_non_faces_files)
label = np.append(label,label_non_faces)
total_data = np.concatenate((data,temp_data),axis=0)
final = np.insert(total_data, num_feature ,label,axis=1)
#final.tofile(base_path + 'train_data.csv',sep=',',format='%10.5f')
pd.DataFrame((final).astype(int)).to_csv(base_path+ "/train_data.csv",header=None, index=None,float_format= '%10.5f')
t1 = time.time()
print((t1-t0)/60)
# test file 
train_faces_files = glob.glob(base_path+ '/dataset/testset/faces/*.png')
train_faces_files.sort()
train_non_faces_files = glob.glob(base_path+ '/dataset/testset/non-faces/*.png')
train_non_faces_files.sort()
data = np.array([[]])
t0 = time.time()
for names in train_faces_files:
    image = cv2.imread(names,cv2.IMREAD_GRAYSCALE)
    i_image = intergal_image(image)
    f = feature_extraction(i_image)
    data = np.append(data,f)

num_image = len(train_faces_files)
num_feature = int(len(data)/num_image)
data = np.resize(data, (num_image,num_feature))
temp_data = np.array([[]])
for names in train_non_faces_files:
    image = cv2.imread(names,cv2.IMREAD_GRAYSCALE)
    i_image = intergal_image(image)
    f = feature_extraction(i_image)
    temp_data = np.append(temp_data,f)
num_image = int(len(temp_data)/num_feature)
temp_data = np.resize(temp_data, (num_image,num_feature))
label = [1]*len(train_faces_files)
label_non_faces = [-1] * len(train_non_faces_files)
label = np.append(label,label_non_faces)
total_data = np.concatenate((data,temp_data),axis=0)
final = np.insert(total_data, num_feature ,label,axis=1)
#final.tofile(base_path + 'train_data.csv',sep=',',format='%10.5f')
pd.DataFrame((final).astype(int)).to_csv(base_path+ "/test_data.csv",header=None, index=None,float_format= '%10.5f')
t1 = time.time()
print((t1-t0)/60)

'''
# test code needs to be deleted 
test  = np.array([1]*(19**2)).reshape(19,19)
feature = []
i_image = intergal_image(test)
image_copy = i_image
feature = feature_extraction(image_copy)
print(len(feature))
'''