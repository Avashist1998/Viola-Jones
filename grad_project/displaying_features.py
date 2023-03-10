import os 
import re
import cv2
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

base_path = os.getcwd()
data = pd.read_csv(base_path+'/10_round_results.csv')
image = cv2.imread((base_path+'/dataset/trainset/faces/face00001.png'),cv2.IMREAD_GRAYSCALE)
feature_names = pd.read_csv(base_path+'/features_names.csv')
for k in range(len((data.J_values)[:])):
    image_with_feature = image.copy()
    string_temp = (feature_names.iloc[(data.J_values)[k]])[0]
    [w,h,i,j] = (re.findall('\d+', string_temp))
    [w,h,i,j] = list(map(int, [w,h,i,j]))
   
    if ((data.J_values).iloc[k] <= 7440):
     #type2
        a = 255*np.array([1]*h*w + [0]*h*w)
        a.resize(h,2*w)
        image_with_feature[i:i+h,j:j+w*2] = a
    elif ( (data.J_values).iloc[k] <= 14880 ):
    #type 1
        a = 255*np.array([1]*h*w + [0]*h*w)
        a.resize(2*h,w)
        image_with_feature[i:i+h*2,j:w+j] = a
    elif ( (data.J_values).iloc[k] <= 18352 ):
    #type 3
        a = 255*np.array([0]*h*w + [1]*2*h*w + [0]*h*w)
        a.resize(4*h,w)
        image_with_feature[i:i+h*4,j:j+w] = a
    elif ( (data.J_values).iloc[k] <= 21824 ):
    #type 4
        a = 255*np.array([0]*h*w + [1]*2*h*w + [0]*h*w)
        a.resize(h,4*w)
        image_with_feature[i:i+h,j:4*w+j] = a
    else:
    #type 5
        a = 255*np.array([1]*h*w + [0]*h*w + [0]*h*w + [1]*h*w)
        a.resize(2*h,2*w)
        image_with_feature[i:i+2*h,j:2*w+j] = a
    
    fig = plt.imshow(image_with_feature,cmap=(plt.cm).gray)
    plt.show(fig)