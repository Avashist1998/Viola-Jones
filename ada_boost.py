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

def decision_stamp(S,Distribution):
    S['Distribution'].loc[:] = Distribution
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

def error_calcuator(prediction,label):
    error = sum(prediction != label)/len(label)
    error_2 =  sum((prediction == -1)& (label == 1))/sum(label == 1)
    error_3 =  sum((prediction == 1)& (label == -1))/sum(label == -1)
    return error,error_2,error_3

def ada_boost(S,y,rounds):
    beta = []
    j_of_round = []
    e_t = [[]]
    theta = []
    new_Distribution = S['Distribution']
    for i in range(0,rounds):
        [J_star,theta_star] = decision_stamp(S,new_Distribution)
        prediction = 2*(S[int(j_star)]>=theta_star).astype(int) - 1 
        [prediction,e1, e2,e3] = error_calcuator(prediction,y)
        [new_Distribution,beta] = weight_cal(Distribution,y,prediction,e1)
        beta.append()
        j_of_round.append(j_star)
        theta.append(theta_star)
        e_t.append(np.array([e1,e2,e3]))
        print(beta,j_of_round,e_t,theta,np.array([e1,e2,e3]))
    return beta, j_of_round, e_t, theta
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
[beta, j_of_round, e_t, theta] = ada_boost(train_df.iloc[:4],train_y.iloc[:4],1)
#[j_star, theta_star] =  decision_stamp(train_X.iloc[:2],Distribution[:2])
print(j_star,theta_star)
print('complete')
print(t1-t0)

