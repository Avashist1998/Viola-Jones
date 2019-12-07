# the goal of the file is to develop the ada_boost algorithm 
import pandas as pd 
import numpy as np 
import os 
import time
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
t0 = time.time()
def beta_cal(epsolon):
    beta = 1/((1-epsolon)/epsolon)
    return beta
def weight_cal(Distribution,label,prediction,error):
    beta = beta_cal(error)
    e = (prediction == label).astype(int)
    Distribution_new = Distribution*beta**e
    Distribution_new = Distribution_new/sum(Distribution_new)
    return Distribution_new,beta

def decision_stamp_search(list_data,F_star,row):
    for i in range(len(list_data)):
        j = list_data[i][0]
        Xj = list_data[i][1]
        Yj = list_data[i][2]
        Dj = list_data[i][3]
        F = sum(Dj[Yj == 1])
        if F< F_star:
            F_star = F
            theta_star = Xj[0]-1
            j_star = j 
        for i in range(0,row-1):
            F = F - Yj[i]*Dj[i]
            if ((F<F_star) &  (Xj[i] != Xj[i+1])):
                F_star = F
                theta_star= 0.5*((Xj[i] + Xj[i+1]))
                j_star=j
    return(j_star,theta_star)
def parllel_sort(S_np,j,row):
    X_np  = S_np[:,:-2]
    Sort = S_np[:,j].argsort()
    Xj =  (S_np[:,j])[Sort]
    Yj =  (S_np[:,-2])[Sort]
    Dj = (S_np[:,-1])[Sort]   
    return(j,Xj,Yj,Dj)
    
def decision_stamp(S,Distribution):
    F_star = float('inf')
    X  = S.drop(columns = ['Label','Distribution'])
    S_np = np.array(S)
    t0 = time.time()
    S_np[:,-1] = Distribution
    print(time.time()-t0)
    [row,col] = S_np.shape
    num_cores = multiprocessing.cpu_count()
    processed_list = Parallel(n_jobs=num_cores,prefer = 'threads')(delayed(parllel_sort)(S_np,j,row) for j in range(col-2))
    print(S_np[:5,-1])
    [j_star,theta_star] = decision_stamp_search(processed_list,F_star,row)
    return (j_star,theta_star)

def error_calcuator(prediction,label):
    error = sum(prediction != label)/len(label)
    error = min(error,1-error)
    error_2 =  sum((prediction == -1)& (label == 1))/len(label)
    error_3 =  sum((prediction == 1)& (label == -1))/len(label)
    return error,error_2,error_3

def ada_boost(S,y,rounds):
    beta_list = []
    j_of_round = []
    e_t = []
    theta = []
    new_Distribution = S['Distribution']
    for i in range(0,rounds):
        print(new_Distribution[:5])
        [J_star,theta_star] = decision_stamp(S,new_Distribution)
        prediction = 2*(S[S.columns[J_star]] < theta_star).astype(int) - 1 
        [e1, e2,e3] = error_calcuator(prediction,y)
        [new_Distribution,beta] = weight_cal(new_Distribution,y,prediction,e1)
        print(new_Distribution[:5])
        print(J_star)
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
train_df = pd.read_csv((base_path+'/Data/train_data.csv'),header = None)
test_df = pd.read_csv((base_path+'/Data/test_data.csv'),header=None)
train_X = train_df.drop(columns = train_df.columns[-1])
train_y = train_df[train_df.columns[-1]]
test_X = train_df.drop(columns = test_df.columns[-1])
test_y = train_df[test_df.columns[-1]]
train_df = df_maker(train_df)
test_df = df_maker(test_df)
print(time.time())
print('time was ', time.time()-t0)
[beta_list, j_of_round, e_t, theta] = ada_boost(train_df[:1000],train_y[:1000],2)
A = pd.DataFrame({'beta':beta_list,'J_values':j_of_round,'theat':theta,'emprical':e_t[:,0],'False Negative':e_t[:,1],'False Positive':e_t[:,2]})
A.to_csv("/Users/abhay/Documents/GitHub/Viola-Jones_Algorithm/10_round_results.csv", index=None,float_format= '%10.5f')
print('time was ', time.time()-t0)
print('complete')