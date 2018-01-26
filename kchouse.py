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
FeatSelect = SelectKBest(f_regression, k=16)
#%%
FeatSelect.fit(X,y)
#%%

KBestFeat = FeatSelect.get_support()

# Eliminates the ID, Condition (probably captured by years since renovation) and the longitude (maybe the latitudinal difference affects the amount of sun that reaches the house)
#%%

#Lets check the f-stat of latitude and longitude

FLong = FeatSelect.scores_[16]
FLat = FeatSelect.scores_[15]

# There is a significant difference between the F-stat for the latitude (2248) and longitude (10). This justifies removing longitude and not latitude.
#%%

# Now for the feature called "condition". We can compare it with the column that represents years since last renovation since it is reasonable to assume that the two would be strongly correlated.
Fcondition = FeatSelect.scores_[8]
FYearsRenov = FeatSelect.scores_[14]

# The F stat values are not decisively different (28 for condition and 61 for years since last renovation) so at this stage we should not really favor one.
#%%
# A few visualizations
# Let's start with a few univariate plots
# Histograms
hpfeatset = dataset.drop(["date","price"],1) # Removing date and price from the original data frame
for i in range(0,21613): # Replacing year of renovation with years since renovation
    if hpfeatset.loc[i,"yr_renovated"] == 0:
        hpfeatset.loc[i,"yr_renovated"] = 2017 - hpfeatset.loc[i,"yr_built"]
    else:
        hpfeatset.loc[i,"yr_renovated"] = 2017 - hpfeatset.loc[i,"yr_renovated"]
hpfeatset = hpfeatset.rename(index=str, columns={"yr_renovated":"yrs_renov"}) # Renaming yr_renovated
hpfeatset.hist(color='blue', alpha=0.5, bins=10) # Plotting the histograms
plt.show()
# The plots show that sqft_lot, sqft_lot15, view and waterfront have almost negligible variation.
#%%
# Making a density plot to get a smooth curve instead of the bins in the histogram to get a better
# sense of the distribution.
hpfeatset.plot(kind='density', subplots=True, layout=(4,5), sharex=False)
# Including these features does not add any value to the model. Also all features besides grade are non-normal
# If we intend to use linear regression we will need to normalize these features. So we can consider eliminating all features with no variation.
#%%
correlations = hpfeatset.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,19,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(hpfeatset.columns.values)
ax.set_yticklabels(hpfeatset.columns.values)
plt.show()
#sqft_above and sqft_living have very high correlation 
#grade and sqft_living have high correlation
#sqft_above and bathrooms have high correlation 
#sqft_living and bathrooms have high correlation 
#sqft_living15 and grade
#sqft_living15 and sqft_above
#The correlation for years_renov and condition is not very high and it makes intuitive sense that the price
# would depend on condition so keeping both.
#%%
# Scatterplot Matrix
import matplotlib.pyplot as plt
import pandas
from pandas.tools.plotting import scatter_matrix
scatter_matrix(hpfeatset)
plt.show()
#%%
FSqftlot = FeatSelect.scores_[4]
FSqftlot15 = FeatSelect.scores_[18]
FView = FeatSelect.scores_[7]
FWaterfront = FeatSelect.scores_[6]
#The F values are low but not as low as longitude and it does not make intuitive sense to eliminate them. So just eliminating ID and longitude.
#%%
