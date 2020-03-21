
import numpy as np
# import pandas as pd
from load_data import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_ols10 = data[['y','x_34','x_14','x_24','x_53','x_ 6','x_51','x_ 3','x_84','x_19','x_16']]
data_ols15 = data[['y','x_34','x_14','x_24','x_53','x_ 6','x_51','x_ 3','x_84','x_19','x_16',
                   'x_ 1','x_52','x_60','x_33','x_68',]]
data_ols20 = data[['y','x_34','x_14','x_24','x_53','x_ 6','x_51','x_ 3','x_84','x_19','x_16',
                   'x_ 1','x_52','x_60','x_33','x_68','x_82','x_43','x_15','x_64','x_61',]]

data_ols10_np = data_ols10.to_numpy()[:,1:]
data_ols15_np = data_ols15.to_numpy()[:,1:]
data_ols20_np = data_ols20.to_numpy()[:,1:]

y = data_ols10_np[:,0]

X10 = data_ols10_np[:,1:]
X15 = data_ols15_np[:,1:]
X20 = data_ols20_np[:,1:]

X10_train, X10_test, y_train, y_test = train_test_split(
    X10, y, test_size=0.10, random_state=42)

X15_train, X15_test, y_train, y_test = train_test_split(
    X15, y, test_size=0.10, random_state=42)

X20_train, X20_test, y_train, y_test = train_test_split(
    X20, y, test_size=0.10, random_state=42)

reg10 = LinearRegression().fit(X10_train, y_train)
y_pred10 = reg10.predict(X10_test)

reg15 = LinearRegression().fit(X15_train, y_train)
y_pred15 = reg15.predict(X15_test)

reg20 = LinearRegression().fit(X20_train, y_train)
y_pred20 = reg20.predict(X20_test)

# root-mean-square error (RMSE)
rmse10 = np.sqrt(np.mean((y_pred10-y_test)**2))
rmse15 = np.sqrt(np.mean((y_pred15-y_test)**2))
rmse20 = np.sqrt(np.mean((y_pred20-y_test)**2))
print('RMSE 10:',np.round(rmse10,4),'\nRMSE 15:',np.round(rmse15,4),'\nRMSE 20:',np.round(rmse20,4))