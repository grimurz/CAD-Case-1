import numpy as np

# X: numpy array
# k_h: half the number of nearest neighbors, i.e. 3 becomes 6-NN
# missing_values: the value to be replaced
# returns a new numpy array
def knn_impute(X, k_h=2, missing_values=None):

    X_nu = X.copy()
    n,m = X.shape
    
    # iterate columns
    for i in np.arange(m):
        
        col = X[:,i]
        
        #iterate values within column
        for j, val in enumerate(col):
            
            # if val == missing_values or np.isnan(val):
            if val == missing_values:
                cnt = 0
                lft = j
                rgt = j
                bag = []
                
                while (lft > 0 or rgt <= n-2) and cnt != 2*k_h:
                    
                    lft_found = False
                    rgt_found = False
                    
                    while lft > 0 and not lft_found:
                        lft -= 1
                        # if col[lft] != missing_values and not np.isnan(col[lft]):
                        if col[lft] != missing_values:
                            bag.append(col[lft])
                            cnt += 1
                            lft_found = True
                            
                    while rgt <= n-2 and not rgt_found:
                        rgt += 1
                        # if col[rgt] != missing_values and not np.isnan(col[rgt]):
                        if col[rgt] != missing_values:
                            bag.append(col[rgt])
                            cnt += 1
                            rgt_found = True
                            
                # print('\nbag',bag)
                # print(np.bincount(bag).argmax(),'\n')
                X_nu[j,i] = np.bincount(bag).argmax()
                        
    return X_nu


# X = np.array([[3, 4, 3], [-1, -1, -1], [3, 4, 3], [-1, 8, 5], [8, 8, 7], [8, 8, 7]])
# test = knn_impute(X, 3, -1)

X = np.array([[3, 4, 3], [-1, np.nan, -1], [3, 4, 3], [-1, 8, 5], [8, 8, 7], [8, 8, 7]])
test = knn_impute(X, 2, None)

