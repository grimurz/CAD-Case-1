
from load_data import X_train, y_train, X_test, y_test #, X_val, y_val

import numpy as np
from tabulate import tabulate

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


N, P = X_train.shape

rf = RandomForestRegressor(
        bootstrap = True,
        oob_score = True,
        random_state = 42
    )

n_estimators = [950] # [int(x) for x in np.linspace(900, 1000, 3)] # [950] #
max_depth = [14] # [int(x) for x in np.linspace(17, 19, 3)] # [14] # [17] # 
min_samples_split = [6] # [int(x) for x in np.linspace(5, 7, 3)] # [6] # 
max_leaf_nodes = [20] #[int(x) for x in np.linspace(17, 23, 3)] # [20] #
min_samples_leaf = [11] # [int(x) for x in np.linspace(10, 12, 3)] # [11] #
# max_features = [120] # [int(x) for x in np.linspace(110, 120, 5)] # [120] # 

param_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'max_leaf_nodes': max_leaf_nodes,
    'min_samples_leaf': min_samples_leaf#,
    # 'max_features': max_features
}


rf_grid = GridSearchCV(
        estimator = rf,
        param_grid = param_grid,
        cv = 5,
        verbose = 2,
        # iid = True,
        n_jobs = -1
    )

# Fit the grid search model
rf_grid.fit(X_train, y_train)

print(rf_grid.best_estimator_)


#%%

## Look at the best estimator and the importance of the features
score = rf_grid.best_estimator_.fit(X_train, y_train)
headers = ["name", "score"]
values = sorted(zip(range(0,P), rf_grid.best_estimator_.feature_importances_), key=lambda x: x[1] * -1)

# See which features are deemed most important by the classifier
# Only gonna look at the 10 most important features out of 256
# print(tabulate(values[0:10], headers[0:10], tablefmt="plain"))
print ('Random Forest OOB error rate: {}'.format(1 - rf_grid.best_estimator_.oob_score_))


#%% Removing least important features improves nothing

# X_train2 = X_train[:, [t[0] for t in values[0:100]]]

# score = rf_grid.best_estimator_.fit(X_train2, y_train)
# headers = ["name", "score"]
# values2 = sorted(zip(range(0,P), rf_grid.best_estimator_.feature_importances_), key=lambda x: x[1] * -1)

# print(tabulate(values2[0:15], headers[0:15], tablefmt="plain"))
# print ('Random Forest OOB error rate: {}'.format(1 - rf_grid.best_estimator_.oob_score_))

#%%

rf = RandomForestRegressor(
        n_estimators = 950,
        max_depth = 14,
        min_samples_split = 6,
        max_leaf_nodes = 20,
        min_samples_leaf = 11,
        # max_features = 120,
        bootstrap = True,
        oob_score = True,
        random_state = 42
    )

# Mean Absolute Error: 4.31
# RMSE: 7.0064

rf.fit(X_train, y_train);

y_pred = rf.predict(X_test)
# y_pred = rf.predict(X_val)

errors = abs(y_pred - y_test)
# errors = abs(y_pred - y_val)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))

# root-mean-square error (RMSE)
rmse = np.sqrt(np.mean((y_pred-y_test)**2))
# rmse = np.sqrt(np.mean((y_pred-y_val)**2))
print('RMSE:',np.round(rmse,4))


