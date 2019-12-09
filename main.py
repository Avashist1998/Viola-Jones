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
data = pd.read_csv(base_path+'/40_round_results.csv')

fig = plt.figure()
ax = plt.axes()
ax.plot(data['round'],data['emprical_total'],label='Empirical Error')
ax.plot(data['round'],data['False Negative_total'],label='False Negative')
ax.plot(data['round'],data['False Positive_total'],label='False Postive')
plt.xlabel('Rounds'), plt.ylabel('Error Percentage')
plt.legend()
plt.title('Rounds vs Error')
plt.show()
data_5_01 = pd.read_csv(base_path+'/10_0.1_round_results.csv')
fig = plt.figure()
ax = plt.axes()
ax.plot(data_5_01['round'],data_5_01['emprical_total'],label='Empirical Error')
ax.plot(data_5_01['round'],data_5_01['False Negative_total'],label='False Negative')
ax.plot(data_5_01['round'],data_5_01['False Positive_total'],label='False Postive')
plt.xlabel('Rounds'), plt.ylabel('Error Percentage')
plt.legend()
plt.title('Rounds vs Error for Gama = 0.1')
plt.show()
data_5_09 = pd.read_csv(base_path+'/10_0.9_round_results.csv')
fig = plt.figure()
ax = plt.axes()
ax.plot(data_5_09['round'],data_5_09['emprical_total'],label='Empirical Error')
ax.plot(data_5_09['round'],data_5_09['False Negative_total'],label='False Negative')
ax.plot(data_5_09['round'],data_5_09['False Positive_total'],label='False Postive')
plt.xlabel('Rounds'), plt.ylabel('Error Percentage')
plt.legend()
plt.title('Rounds vs Error for Gama = 0.9')
plt.show()