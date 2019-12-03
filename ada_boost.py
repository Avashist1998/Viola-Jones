# the goal of the file is to develop the ada_boost algorithm 
import pandas as pd 
import numpy as np 
import os 
import time
import matplotlib.pyplot as plt
def beta_cal(epsolon):
    beta = 1/((1-epsolon)/epsolon)
    return beta
def weight_cal(Distribution,beta,label,prediction):
    e = int(prediction == label)
    Distribution_new = Distribution*beta**(1-e)
    return Distribution_new

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

def error_calcuator(kind,S,threshold,J):
    prediction = int(S[J] >= threshold)
    label = S['Label']
    if (kind == 1):
        error = sum(prediction != label)/len(label)
    elif (kind == 2):
        error_2 =  sum((prediction == -1)& (label == 1))/sum(label == 1)
    else:
        error_3 =  sum((prediction == 1)& (label == -1))/sum(label == -1)
    return prediction,error,error_2,error_3

def ada_boost(S,rounds):

    alpha = np.array([0]*round)
    e_t = np.round([0]*round)
    for i in rounds:
        print(i)
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
t0 = time.time()
[j_star, theta_star] =  decision_stamp(train_X,Distribution)
t1 = time.time()
print(j_star,theta_star)
print('complete')
print(t1-t0)