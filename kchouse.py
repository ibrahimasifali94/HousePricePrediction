#Importing the Libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%

#Importing the Dataset 
dataset = pd.read_csv('kc_house_data.csv')

#%%
# Separated the labels/prices
y = dataset["price"].get_values()
#%%
hprice = dataset.get_values()
#%%
# Get the feature set
X = np.delete(hprice,2,1)
#%%
# Replacing last renovated year in Col 15 with the years since last renovation
for i in range(0,21613):
    if X[i,14] == 0:
        X[i,14] = 2017 - X[i,13]
    else:
        X[i,14] = 2017 - X[i,14]
#%%
# Deleting date column i.e. the 2nd column
X = np.delete(X,1,1)
#%%    
# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import GenericUnivariateSelect
#%%
FeatSelect = SelectKBest(f_regression, k=17)
#%%
FeatSelect.fit(X,y)
#%%


