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
def weight_cal(Distribution,label,prediction):
    e = (prediction == label).astype(int)
    epsolan = sum(Distribution*(1-e))
    beta = beta_cal(epsolan)
    Distribution_new = Distribution*beta**e
    Distribution_new = Distribution_new/sum(Distribution_new)
    return Distribution_new,beta

def decision_stamp_search(list_data,row):
    F_star = float('inf')
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
    [row,col] = S_np.shape
    num_cores = multiprocessing.cpu_count()
    processed_list = Parallel(n_jobs=num_cores,prefer = 'threads')(delayed(parllel_sort)(S_np,j,row) for j in range(col-2))
    [j_star,theta_star] = decision_stamp_search(processed_list,row)
    return (j_star,theta_star)

def error_calcuator(prediction,label):
    error = sum(prediction != label)/len(label)
    error_2 =  sum((prediction == -1)& (label == 1))/len(label)
    error_3 =  sum((prediction == 1)& (label == -1))/len(label)
    return error,error_2,error_3

def ada_boost(S,y,rounds):
    beta_list = []
    j_of_round = []
    e_t = []
    theta = []
    parity_tol = []
    new_Distribution = S['Distribution']
    for i in range(0,rounds):
        [J_star,theta_star] = decision_stamp(S,new_Distribution)
        prediction = 2*(S[S.columns[J_star]]>=theta_star).astype(int) - 1
        [e1, e2,e3] = error_calcuator(prediction,y)
        if e1 <= 0.5:
            parity = 1
        else :
            parity = -1
            prediction = 2*(S[S.columns[J_star]]<=theta_star).astype(int) - 1
        parity_tol.append(parity)
        [e1, e2,e3] = error_calcuator(prediction,y)
        [new_Distribution,beta] = weight_cal(new_Distribution,y,prediction,e1)
        beta_list.append(beta)
        j_of_round.append(J_star)
        theta.append(theta_star)
        e_t.append([e1,e2,e3])
    e_t = np.array(e_t)
    e_t.resize(rounds,3)
    return (np.array(beta_list), np.array(j_of_round), np.array(e_t), np.array(theta), np.array(parity_tol))
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
def total_error(test_X,test_y,theta,parity_tol):
    test_np_X = np.array(test_X)
    test_np_y = np.array(test_y)
    test_np_X[:,j_of_round[:]]
    thresholed_total = (np.multiply(parity_tol,test_np_X[:,j_of_round[:]]) < np.multiply(1,theta)).astype(int)
    predictions_total = []
    combined_error = []
    alpha = np.log(1/np.array(beta_list))
    for i in range(1,len(alpha)+1):
        prediction_t = (np.dot(thresholed_total[:,:i],alpha[:i]) >= 0.5*sum(alpha[:i])).astype(int)
        predictions_total.append(prediction_t)
    predictions_total = np.array(predictions_total)
    sum_error = []
    for i in range(len(alpha)):
        sum_error.append(error_calcuator(2*predictions_total[i]-1,test_np_y))
    sum_error = np.array(sum_error)
    return(sum_error)





base_path  =  os.getcwd()
rounds = 10
train_df = pd.read_csv((base_path+'/Data/train_data.csv'),header = None)
test_df = pd.read_csv((base_path+'/Data/test_data.csv'),header=None)
train_X = train_df.drop(columns = train_df.columns[-1])
train_y = train_df[train_df.columns[-1]]
test_X = train_df.drop(columns = test_df.columns[-1])
test_y = train_df[test_df.columns[-1]]
train_df = df_maker(train_df)
test_df = df_maker(test_df)
[beta_list, j_of_round, e_t, theta,parity_tol] = ada_boost(train_df,train_y,rounds)
features_names = pd.read_csv(base_path+'/features_names.csv',header = None)
J_names = (features_names.iloc[j_of_round])[0]
sum_error = total_error(test_X,test_y,theta,parity_tol)
A = pd.DataFrame({'round':np.arange(1, rounds+1, 1),'beta':beta_list,'J_values':j_of_round,'theat':theta,'emprical':e_t[:,0],'False Negative':e_t[:,1],'False Positive':e_t[:,2],'pairty':parity_tol,'emprical_total':sum_error[:,0],'False Negative_total':sum_error[:,1],'False Positive_total':sum_error[:,2]})
A.to_csv('/Users/abhay/Documents/GitHub/Viola-Jones_Algorithm/'+ str(rounds)+'_round_results.csv', index=None,float_format= '%10.5f')
print('complete')
