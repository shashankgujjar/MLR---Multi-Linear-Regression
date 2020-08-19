# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:47:26 2020

@author: user
"""

#Consider only the below columns and prepare a prediction model for predicting Price.
#Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]

import pandas as pd
toyota_corolla = pd.read_csv('E:\\Data\\Assignments\\i made\\MLR\\ToyotaCorolla.csv' , engine = 'python')
toyota_corolla.columns

# Creating a data frame
# considering only required columns
# Dropping unwanted columns
toyota = pd.DataFrame(toyota_corolla, columns = ["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"])

toyota.columns
toyota.describe()
toyota.shape

corr = toyota.corr()

#Generate Heat Map, allow annotations and place floats in map
import seaborn as sns
sns.heatmap(corr, cmap='magma', annot=True, fmt=".2f")

# to check the collinearity btwn inputs

import seaborn as sns
sns.pairplot(toyota)

# Model Building
import statsmodels.formula.api as smf # for regression model

# Model 1

model1 = smf.ols('Price~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data=toyota).fit()
model1.summary() # R-sqr= 0.864
# p values of cc and Doors are high 

# constructing models on individual basis
# bcz of hivh pvlue of cc and doors
modelcc = smf.ols('Price~ cc', data=toyota).fit()
modelcc.summary() # p-val 0 # significant

modelDoors = smf.ols('Price~ Doors', data=toyota).fit()
modelDoors.summary() # p-val 0  # significant

modelccDoors = smf.ols('Price~ Doors+cc', data=toyota).fit()
modelccDoors.summary() # p-val 0
# significant


# Influece plot
import statsmodels.api as sm
sm.graphics.influence_plot(modelccDoors)
influence.measures(modelccDoors.toyota)

# Removing Influencing Data points
toyota_new = toyota.drop(toyota.index[[80]], axis=0)
print(toyota_new.shape)

# Model 2: based on removed influencing data points

model2 = smf.ols('Price~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data=toyota_new).fit()
model2.summary() # R-sqr= 0.869
# p value of Doors still 0.488 

# Partial regressor plot
sm.graphics.plot_partregress_grid(model2)
# From above plot,
# the most straight line having variable will be removed
# as it is insignificant

# Model 3 

model3 = smf.ols('Price~ Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight', data=toyota_new).fit()
model3.summary() # R-sqr= 0.869
# p values of all the variables are less than 0.05


# Predicted values of Price 
price_pred = model3.predict(toyota_new)
price_pred

actual = toyota_new['Price']
predicted = price_pred

Error = predicted - actual

# RMSE

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_mlR3 = sqrt(mean_squared_error(actual, predicted))
print(rmse_mlR3) # 1308.7117736695945


