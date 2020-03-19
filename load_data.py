
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from knn_impute import knn_impute

data = pd.read_csv('./Case Data/case1Data.txt', delimiter = ", ")


data['C_ 1'] = pd.Categorical(data['C_ 1']).codes
data['C_ 2'] = pd.Categorical(data['C_ 2']).codes
data['C_ 3'] = pd.Categorical(data['C_ 3']).codes
data['C_ 4'] = pd.Categorical(data['C_ 4']).codes
data['C_ 5'] = pd.Categorical(data['C_ 5']).codes


# for i in range(1,6):
#     col = 'C_ ' + str(i)    
#     temp_df = pd.get_dummies(data[col])
#     temp_df = temp_df.add_suffix(str(i))
#     data = pd.concat([data, temp_df], axis=1)
#     data = data.drop(columns=col)


data_cate_imp = knn_impute(data.to_numpy()[:,96:101], 4, -1)
df_cate_imp = pd.DataFrame(
        data = data_cate_imp,
        columns = ['C_ 1','C_ 2','C_ 3','C_ 4','C_ 5']
    )

for i in range(1,6):
    col = 'C_ ' + str(i)    
    temp_df = pd.get_dummies(df_cate_imp[col])
    temp_df = temp_df.add_prefix(str(i)+' ')
    data = pd.concat([data, temp_df], axis=1)
    data = data.drop(columns=col)

data_np = data.to_numpy()


X = data_np[:,1:]
y = data_np[:,0]

# X_train, X_temp, y_train, y_temp = train_test_split(
#     X, y, test_size=0.20, random_state=42)

# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.50, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)


# https://inmachineswetrust.com/posts/drop-first-columns/

# scikit-learn uses an optimised version of the CART algorithm;
# however, scikit-learn implementation does not support categorical variables for now.

# test1 = data['C_ 1'].value_counts()
# test2 = data['C_ 2'].value_counts()
# test3 = data['C_ 3'].value_counts()
# test4 = data['C_ 4'].value_counts()
# test5 = data['C_ 5'].value_counts()

# del X_temp, y_temp
# del temp_df, i, col
# del data, data_np
