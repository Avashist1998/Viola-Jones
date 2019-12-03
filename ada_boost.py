# the goal of the file is to develop the ada_boost algorithm 
import pandas as pd 
import numpy as np 
import os 
import time
import matplotlib.pyplot as plt
base_path  =  os.getcwd()
train_df = pd.read_csv((base_path+'/Data/train_data.csv',header=None)
test_df = pd.read_csv((base_path+'/Data/test_data.csv',header=None)
