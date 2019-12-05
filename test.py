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
    name = []
    [row, col] = image_copy.shape
    # Type 1 features (two vertical)
    for w in range(1,5):
        for h in range(1,9):
            for i in range(row-h+1): 
                for j in range(col-2*w+1):
                    output = -2*image_copy[i+h-1,j+w-1] + 2*image_copy[i,j+w-1] + image_copy[i+h-1,j+2*w-1] + image_copy[i+h-1,j] + image_copy[i,j+2*w-1] - image_copy[i,j]
                    feature.append(output)
                    name.append('w = '+str(w)+' h = '+str(h)+' i = ' + str(i)+ ' j = ' + str(j))
    print(len(feature))
    #Type 2 features (two horizontal)
    for h in range(1,5):
        for w in range(1,9):
            for i in range(row-2*h+1): 
                for j in range(col-w+1):
                    output = 2*image_copy[i+h-1,j] + image_copy[i+2*h-1,j+w-1] + image_copy[i,j+w-1] - 2*image_copy[i+h-1,j+w-1] - image_copy[i+2*h-1,j] - image_copy[i,j]
                    feature.append(output)
                    name.append('w = '+str(w)+' h = '+str(h)+' i = ' + str(i)+ ' j = ' + str(j))
    print(len(feature))
    # Type 3 feature (three Horizonatal)
    for h in range(1,3):
        for w in range(1,9):
            for i in range(row-4*h+1): 
                for j in range(col-w+1):
                    output = 2*image_copy[i+3*h-1,j+w-1] + 2*image_copy[i+h-1,j] - 2*image_copy[i+h-1,j+w-1] - 2*image_copy[i+3*h-1,j] - image_copy[i+4*h-1,j+w-1] + image_copy[i+4*h-1,j] - image_copy[i,j] + image_copy[i,j+w-1]
                    feature.append(output)
                    name.append('w = '+str(w)+' h = '+str(h)+' i = ' + str(i)+ ' j = ' + str(j))
    print(len(feature))
    # Type 4 feature (two Vertical)
    for h in range(1,9):
        for w in range(1,3):
            for i in range(row-h+1): 
                for j in range(col-4*w+1):
                    output = 2*image_copy[i,j+w-1] + 2*image_copy[i+h-1,j+3*w-1] - 2*image_copy[i,j+3*w-1] - 2*image_copy[i+h-1,j+w-1] - image_copy[i,j]+ image_copy[i+h-1,j] - image_copy[i+h-1,j+4*w-1] + image_copy[i,j+4*w-1]
                    feature.append(output)
                    name.append('w = '+str(w)+' h = '+str(h)+' i = ' + str(i)+ ' j = ' + str(j))
    print(len(feature))
    # Type 5 feature (four)
    for h in range(1,5):
        for w in range(1,5):
            for i in range(row-2*h+1): 
                for j in range(col-2*w+1):
                    output = image_copy[i,j]+ 4*image_copy[i+h-1,j+w-1] - 2*image_copy[i,j+w-1] - 2*image_copy[i+h-1,j] + image[i+2*h-1,j+2*w-1] - 2*image_copy[i+h-1,j+2*w-1] + image_copy[i,j+2*w-1] - 2*image_copy[i+2*h-1,j+w-1] + image_copy[i+2*h-1,j]
                    feature.append(output)
                    name.append('w = '+str(w)+' h = '+str(h)+' i =' + str(i)+ ' j =' + str(j))
    print(len(feature))
    return feature, name
'''
def decision_stamp(S,Distribution):
    col = []
    for names in S:
        col.append(str(names))
    col[-1] = 'Label'
    col = col + ['Distribution']
    S = S.merge(pd.Series(Distribution).to_frame(), left_index=True, right_index=True)
    S.columns = col 
    F_star = float('inf')
    X  = S.drop(columns = ['Label','Distribution'])
    for col in X.columns:
        X  = S.drop(columns = ['Label','Distribution'])
        Xj = S.sort_values(by = col)
        Xj = Xj.reset_index(drop=True)
        Yj = Xj['Label']
        Dj = Xj['Distribution'] 
        Xj = Xj.drop(columns = Xj.columns[col != Xj.columns])
        F = sum(Dj[Yj == 1])
        if F< F_star:
            F_star = F
            theta_star = (Xj.iloc[0])[0] -1 
            j_star = col  
        for i in range(0,len(Xj)-1):
            F = F - Yj.iloc[i]*Dj.iloc[i]
            if ((F<F_star) &  (Xj.iloc[i] != Xj.iloc[i+1])[0]):
                F_star = F
                theta_star= 0.5*((Xj.iloc[i] + Xj.iloc[i+1])[0])
                j_star=col
    return(j_star,theta_star)
base_path  =  os.getcwd()
train_df = pd.read_csv('Data/train_data.csv',header = None)
test_df = pd.read_csv('Data/test_data.csv',header=None)
print('done')
train_X = train_df.drop(columns = train_df.columns[-1])
train_y = train_df[train_df.columns[-1]]
test_X = train_df.drop(columns = test_df.columns[-1])
test_y = train_df[test_df.columns[-1]]
P_count = train_y[train_y == 1].count() 
N_count = len(train_y) - P_count
Distribution = np.array([1/(2*P_count)]*P_count + [1/(2*N_count)]*N_count)
t0 = time.time()
[j_star, theta_star] =  decision_stamp(train_X.iloc[:10],Distribution[:10])
t1 = time.time()
print(j_star,theta_star)
'''
base_path  =  os.getcwd()
a = np.array([1]*(19*19))
a.resize(19,19)
i_image = intergal_image(a)
[feature, names] = feature_extraction(i_image)
pd.DataFrame(names).to_csv(base_path+ "/features_names.csv",header=None, index=None)
print(names[14227])
print('complete')