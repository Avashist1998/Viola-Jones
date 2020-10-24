import numpy as np
import multiprocessing
from joblib import Parallel,delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DecisionStump():

    def __init__(self):
        # Determines if sample shall be classified as -1 or 1 given threshold
        self.polarity = 1
        # The index of the feature used to make classification
        self.feature_index = None
        # The threshold value that the feature should be measured against
        self.threshold = None
        # Value indicative of the classifier's accuracy
        self.alpha = None
        self.empirical_error = None
        self.val_empirical_error= None
        self.false_neg_error = None
        self.val_false_neg_error = None
        self.false_pos_error = None
        self.val_false_pos_error = None
        self.weight_error = None
        self.weights = None


# Assuming that y is in turm of -1 and 1 

class ada_boost():
    def __init__(self,n_estimators = 10):
        self.n_estimators = n_estimators
    
    def get_params(self, deep = True):
        return{'n_estimators':self.n_estimators}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    
    def parllel_sort(self,X,y,weights,j):
        Sort = X[:,j].argsort()
        Xj =  (X[:,j])[Sort]
        Yj =  (y[:])[Sort]
        Dj = (weights[:])[Sort]   
        return(j,Xj,Yj,Dj)

    def decision_stamp_search(self,list_data,row):
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


    def beta_cal(self,epsolon):
        beta = 1/((1-epsolon)/epsolon)
        return beta

    def weight_cal(self,Distribution,label,prediction):
        e = (prediction == label).astype(int)
        epsolan = np.sum(Distribution*(1-e))
        beta = self.beta_cal(epsolan)
        if epsolan == 0:
            Distribution_new = Distribution
            beta = 0.00001
        else:
            Distribution_new = Distribution*beta**e
            Distribution_new = Distribution_new/sum(Distribution_new)
        alpha = np.log(1/beta)
        return Distribution_new,alpha


    def fit(self, X,y,sample_weight = None, validation_data = None,validation_percentage = None):
        no_val_data_flag=False
        if validation_percentage != None and validation_data == None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_percentage,shuffle=True)
        elif validation_data != None:
            X_val, y_val= validation_data
            X_train, y_train = X,y
        else:
            X_train, y_train = X, y
            no_val_data_flag = True
            
        if sample_weight == None:
            labels, counts = np.unique(y_train, return_counts=True)
            counts = 0.5*(1/np.array(counts))
            #weights = np.full(n_samples, 1/n_samples)
            new_int_weights=[]
            for i in range(len(y_train)):
                if y_train[i] == labels[0]:
                    new_int_weights.append(counts[0])
                else:
                    new_int_weights.append(counts[1])
            weights = np.array(new_int_weights)
        else:
            weights = sample_weight
        self.base_clfs = []
        for _ in range(self.n_estimators):
            clf = DecisionStump()
            [row,col] = X_train.shape
            num_cores = multiprocessing.cpu_count()
            processed_list = Parallel(n_jobs=num_cores)(delayed(self.parllel_sort)(X_train,y_train,weights,j) for j in range(col))
            [j_star,theta_star] = self.decision_stamp_search(processed_list,row)
            p = 1
            prediction = np.ones(np.shape(y_train))
            prediction[X_train[:,j_star] < theta_star] = -1
            error = np.sum(weights*(y_train != prediction).astype(int))
            if error > 0.5:
                error = 1 - error
                p = -1
                prediction = -1*prediction
            clf.weight_error = error
            clf.weights = weights
            clf.polarity= p
            clf.threshold = theta_star
            clf.feature_index = j_star
            weights, clf.alpha = self.weight_cal(weights, y_train, prediction)
            self.base_clfs.append(clf)
            prediction =  self.predict(X_train)
            clf.empirical_error = sum(prediction != y_train)/len(y_train)
            clf.false_neg_error = sum((prediction == -1)& (y_train == 1))/len(y_train)
            clf.false_pos_error = sum((prediction == 1)& (y_train == -1))/len(y_train)
            if no_val_data_flag != True:
                prediction =  self.predict(X_val)
                clf.val_empirical_error = sum(prediction != y_val)/len(y_val)
                clf.val_false_neg_error = sum((prediction == -1)& (y_val == 1))/len(y_val)
                clf.val_false_pos_error = sum((prediction == 1)& (y_val == -1))/len(y_val)

            
    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        for clf in self.base_clfs:
            # Set all predictions to '1' initially
            predictions = np.ones(np.shape(y_pred))
            # The indexes where the sample values are below threshold
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # Label those as '-1'
            predictions[negative_idx] = -1
            # Add predictions weighted by the classifiers alpha
            # (alpha indicative of classifier's proficiency)
            y_pred += clf.alpha * predictions
        # Return sign of prediction sum
        y_pred = np.sign(y_pred).flatten()
        return y_pred

    def score(self, X, y):
        X_prediction = self.predict(X)
        return(accuracy_score(y_true = y, y_pred= X_prediction))