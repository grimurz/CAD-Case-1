
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_data import data_non_1ook

y_corr = data_non_1ook.corr()['y']

y_corr_abs = abs(y_corr)

# http://benalexkeen.com/correlation-in-python/

plt.matshow(data_non_1ook.corr())
plt.xticks(range(len(data_non_1ook.columns)), data_non_1ook.columns)
plt.yticks(range(len(data_non_1ook.columns)), data_non_1ook.columns)
plt.colorbar()
plt.show()

