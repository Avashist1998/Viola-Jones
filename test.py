# the goal of the file is to develop the ada_boost algorithm 
import pandas as pd 
import numpy as np 
import os 
import time
import matplotlib.pyplot as plt

def beta_cal(epsolon):
    beta = 1/((1-epsolon)/epsolon)
    return beta
def weight_cal(Distribution,label,prediction,error):
    beta = beta_cal(error)
    e = (prediction == label).astype(int)
    Distribution_new = Distribution*beta**(1-e)
    Distribution_new = Distribution_new/sum(Distribution_new)
    return Distribution_new,beta

def decision_stamp(S_orignal,Distribution):
    S = S_orignal.copy()
    (S['Distribution'].loc[:])[:]  = Distribution
    F_star = float('inf')
    X  = S.drop(columns = ['Label','Distribution'])
    X_np = np.array(X)
    S_np = np.array(S)
    [row,col] = X_np.shape
    for j in range(col):
        X_np  = S_np[:,:-2]
        X_np_j = S_np[S_np[:,j].argsort()]
        Yj =  X_np_j[:,-2]
        Dj =  X_np_j[:,-1]
        Xj =  X_np_j[:,j]
        F = sum(Dj[Yj == 1])
        if F< F_star:
            F_star = F
            theta_star = Xj[0]-1
            j_star = col  
        for i in range(0,row-1):
            F = F - Yj[i]*Dj[i]
            if ((F<F_star) &  (Xj[i] != Xj[i+1])):
                F_star = F
                theta_star= 0.5*((Xj[i] + Xj[i+1]))
                j_star=j
    return(j_star,theta_star)

def error_calcuator(prediction,label):
    error = sum(prediction != label)/len(label)
    #False negative
    error_2 =  sum((prediction == -1)& (label == 1))/len(label)
    #False Positive
    error_3 =  sum((prediction == 1)& (label == -1))/len(label)
    return error,error_2,error_3

def ada_boost(S,y,rounds):
    beta_list = []
    j_of_round = []
    e_t = []
    theta = []
    new_Distribution = S['Distribution']
    for i in range(0,rounds):
        [J_star,theta_star] = decision_stamp(S,new_Distribution)
        prediction = 2*(S[S.columns[J_star]]>=theta_star).astype(int) - 1 
        [e1, e2,e3] = error_calcuator(prediction,y)
        [new_Distribution,beta] = weight_cal(new_Distribution,y,prediction,e1)
        beta_list.append(beta)
        j_of_round.append(J_star)
        theta.append(theta_star)
        e_t.append([e1,e2,e3])
    e_t = np.array(e_t)
    e_t.resize(rounds,3)
    return beta_list, j_of_round, e_t, theta
def df_maker(S):
    col = []
    for names in S.columns:
        col.append(str(names))
    col[-1] = 'Label'
    col = col + ['Distribution']
    train_y = S[S.columns[-1]]
    P_count = train_y[train_y == 1].count() 
    N_count = len(train_y) - P_count
    Distribution = np.array([1/(2*P_count)]*P_count + [1/(2*N_count)]*N_count)
    S = S.merge(pd.Series(Distribution).to_frame(), left_index=True, right_index=True)
    S.columns = col
    return S
base_path  =  os.getcwd()
train_df= pd.read_csv((base_path+'/Data/train_data.csv'),header = None)

train_X = train_df.drop(columns = train_df.columns[-2:])
train_y = train_df[train_df.columns[-2]]
print(train_y)
t0  = time.time()
[beta_list, j_of_round, e_t, theta] = ada_boost(train_df,train_y,1)

#[j_star, theta_star] =  decision_stamp(train_X.iloc[:2],Distribution[:2])
print('Time is = ',time.time()-t0)
