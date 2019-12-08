import pandas as pd 
import numpy as np 
import os 
import time
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import seaborn as sns
import cv2 
base_path = os.getcwd()
data_0.05 = pd.read_csv(base_path+'/10_0.5_round_results.csv')
data_0.08 = pd.read_csv(base_path+'/10_0.8_round_results.csv')