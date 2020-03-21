# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:39:23 2020

@author: P
"""
import scipy.io
import numpy as np
from sklearn import linear_model 
from scipy import linalg
from sklearn import preprocessing 
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from scipy.stats import linregress
from statsmodels.sandbox.stats.multicomp import multipletests 
import warnings # to silence convergence warnings
import os
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
sns.set() # Set searborn as default
import pandas as pd
from scipy . spatial import distance # 
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
# %% Run this to set all NaN to the zero-vector
dataPath = os.path.join('Case Data', 'case1Data.txt')
data = pd.read_csv(dataPath, sep=", ", header=0)
y = data['y'].to_numpy()[:,None]
X = pd.get_dummies(data, columns=["C_ 1", "C_ 2","C_ 3","C_ 4","C_ 5"]).to_numpy()
X = X[:,1:]
#%%
# Run this if you want to drop all NaN
indexes =np.unique(np.where(data.isna())[0])
data_drop = data.drop(indexes)
X = pd.get_dummies(data_drop, columns=["C_ 1", "C_ 2","C_ 3","C_ 4","C_ 5"]).to_numpy()[:,1:]
y = data_drop['y'].to_numpy()[:,None]

#%% KNN
K = 3
kf = KFold(n_splits=K)

with warnings.catch_warnings(): # done to disable all the convergence warnings from elastic net
    warnings.simplefilter("ignore")
    num_neigh = 10
    Error = np.zeros((K, num_neigh))
    
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
    
        for k in range(1,num_neigh+1):
            # Use Scikit KNN classifier, as you have already tried implementing it youself
            neigh = KNeighborsRegressor(n_neighbors=k, weights = 'uniform')
            neigh.fit(X_train, y_train)
            yhat = neigh.predict(X_test)
                
            # This time i use the MAE
            Error[i-1, k-1] = np.mean((y_test - yhat[:,None])**2)
              
    E = np.mean(Error, axis = 0)
        
#%% ElasticNet
K = 3
kf = KFold(n_splits=K)
## Add intercept
Xc = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
with warnings.catch_warnings(): # done to disable all the convergence warnings from elastic net
    warnings.simplefilter("ignore")
    num_neigh = 10    
    alphas = [0.01, 0.1, 1, 10, 100]
    Error = np.zeros((K, num_neigh,len(alphas)))
    
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train = Xc[train_index]
        y_train = y[train_index]
        X_test = Xc[test_index]
        y_test = y[test_index]
        
        for k in range(1,num_neigh+1):
            for a in range(len(alphas)):
            # Use Scikit KNN classifier, as you have already tried implementing it youself
                eNet = ElasticNet(l1_ratio = k/num_neigh,alpha=alphas[a])
                eNet.fit(X_train, y_train)
                yhat = eNet.predict(X_test)
                    
                # This time i use the MAE
                Error[i-1, k-1,a] = np.mean((y_test - yhat[:,None])**2)
              
    E = np.mean(Error, axis = 0)
E

#%%
## See the fit on a plot for a random test-set
eNet = ElasticNet(l1_ratio = 3/num_neigh,alpha=alphas[2])
eNet.fit(X_train, y_train)
yhat = eNet.predict(X_test)
plt.plot(yhat)
plt.plot(y_test)

