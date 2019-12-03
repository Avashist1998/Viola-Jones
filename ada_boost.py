# the goal of the file is to develop the ada_boost algorithm 
import pandas as pd 
import numpy as np 
import os 
import time
import matplotlib.pyplot as plt
def alpha_cal(epsolon):
    alpha = 0.5 * ln((1-epsolon)/epsolon)
    return alpha
def weight_cal(w,z,alpha,h,y):
    w_new = w/z
    return w_new

def decision_stamp(S,Distribution)

    col = []
    for names in S:
        col.append(str(names))
    col[-1] = 'Label'
    col = col + ['Distribution']
    S = S.merge(pd.Series(Distribution).to_frame(), left_index=True, right_index=True)
    S.columns = col 
    F_star = float('inf')
    for col in X.columns:
        X  = S.drop(columns = ['Label','Distribution'])
        Xj = S.sort_values(by = col)
        Xj = Xj.reset_index(drop=True)
        Yj =  Xj['Label']
        Dj =  Xj['Distribution'] 
        Xj =  Xj.drop(columns = Xj.columns[col != Xj.columns])
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
def ada_boost(S,rounds):
    alpha = np.array([0]*round)
    e_t = np.round([0]*round)
    for i in rounds:
        
    return(12)
base_path  =  os.getcwd()
train_df = pd.read_csv((base_path+'/Data/train_data.csv'),header = None)
test_df = pd.read_csv((base_path+'/Data/test_data.csv'),header=None)
train_X = train_df.drop(columns = train_df.columns[-1])
train_y = train_df[train_df.columns[-1]]
test_X = train_df.drop(columns = test_df.columns[-1])
test_y = train_df[test_df.columns[-1]]
P_count = train_y[train_y == 1].count() 
N_count = len(train_y) - P_count
Distribution = np.array([1/(2*P_count)]*P_count + [1/(2*N_count)]*N_count)

print('complete')
