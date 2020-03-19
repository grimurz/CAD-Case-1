
from load_data import X_train, y_train, X_test, y_test #, X_val, y_val

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

N, P = X_train.shape


n_estimators = [int(x) for x in np.linspace(200, 1000, 5)] # [] #
learning_rate = [float(x) for x in np.linspace(0.01, 0.9, 3)]  # [0.01] #

ab = AdaBoostRegressor(
        base_estimator = DecisionTreeRegressor(max_depth=3),
        random_state = 42
    )
                      
param_grid = {
    'n_estimators': n_estimators,
    'learning_rate': learning_rate
}

ad_grid = GridSearchCV(
        estimator = ab, 
        param_grid = param_grid, 
        cv = 5, 
        verbose=2, 
        # iid = True, 
        n_jobs = -1
    )

# Fit the grid search model
ad_grid.fit(X_train, y_train)

boost_score = ad_grid.cv_results_['mean_test_score'].reshape(len(learning_rate),
                                                     len(n_estimators))

plt.figure()
for ind, i in enumerate(learning_rate):
    plt.plot(n_estimators, boost_score[ind], label='Learning Rate: {0:.2f}'.format(i))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Mean accuracy')
plt.show()