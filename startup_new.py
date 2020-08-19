#........load data........
import pandas as pd
startup = pd.read_csv("E:\\Data\\Assignments\\i made\\MLR\\50_Startups.csv")
startup.columns
print(startup.shape)

#.....EDA...........
#...Business moments 1....
startup.mean()
startup.median()

startup.describe()

startup.var()
startup.std()

#....Outlier presence Business moments 2 ....(BOX-PLOTS)
import matplotlib.pyplot as plt
plt.boxplot(startup["R_and_D_Spend"])
plt.boxplot(startup["Administration"])
plt.boxplot(startup["Marketing_Spend"])
plt.boxplot(startup["Profit"])

# Checking Whether data is normally distributed
import pylab         
import scipy.stats as st

st.probplot(startup['R_and_D_Spend'], dist="norm",plot=pylab)
st.probplot(startup['Administration'], dist="norm",plot=pylab)
st.probplot(startup['Marketing_Spend'], dist="norm",plot=pylab)
st.probplot(startup['Profit'], dist="norm",plot=pylab)


# Removing State column since it is Categorical
# Before Removing we need to Crete A DataFrame
df = pd.DataFrame(startup)
print(df.shape)

df_drop_state = df.drop(["State"], axis=1 ) #Removing
print(df_drop_state.shape)

# Transformations 
#import numpy as np
#pnew = np.log(startup["Profit"])
#pnew

#there is no need to transform since the data is 
#following normal distribution


# Normalize Or Standardising Data
# To normalize from scikitlearn 
# (Normalize Range :0 to 1, Standardise Range :-3.4 to +3.4) 
# To make Data scale free and unitless

#X = df_drop_state
#from sklearn import preprocessing
#normalized_X = preprocessing.normalize(X)

#..........End of EDA.........


# Creating dummies of State column
dummy = pd.get_dummies(startup["State"])

# combining Dummy with normalized data
new = pd.concat([df_drop_state, dummy], axis=1)
new
#
##....Combining with original data which doesnt have state column.. NON normalized
#Data_with_Dummy = pd.concat([df_drop_state, dummy], axis=1)
#
#print(Data_with_Dummy.shape)

## After creating dummies, adding dummies to Data Frame 
#df = pd.concat([df,dummy],axis=1)
#print(df.shape)

# Deleting or Dropping a column using pandas module,
# since we have dummies, there is no need of *State* column in data set
#df_new = df.drop(["State"], axis=1 )
#print(df_new.shape)

# OR to avoid deleting state column from data frame again
# we can use, directly
# df = pd.concat([df_drop_state,dummy],axis=1)


#Checking the existatnce of Collinearity between input variables....

import seaborn as sns

sns.pairplot(df_drop_state)

df_drop_state.corr()

# model building
import statsmodels.formula.api as smf # for regression model

ml1 = smf.ols('Profit~R_and_D_Spend+Administration+Marketing_Spend',data=new).fit() # regression model
ml1.params
ml1.summary()

new.corr()

# p-values for Marketing_Spend are more than 0.05 
#so need to check which influencing more 

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

# From influence plot, 49, 50 influencing more
# dropping those 
rmv_influenced = new.drop(new.index[[48,49]],axis=0)

# Preparing model with Removed Influenced data points
ml_2=smf.ols('Profit~R_and_D_Spend+Administration+Marketing_Spend',data = rmv_influenced).fit()  
ml_2.summary()



# preparing model with single input, significant variable will be kept
#R_and_D_Spend
RD_ml = smf.ols('Profit~R_and_D_Spend',data = rmv_influenced).fit()
RD_ml.summary()
#p value and R sqrd value is 0.958
#significant


# Market Spend
MS_ml = smf.ols('Profit~Marketing_Spend',data = rmv_influenced).fit()
MS_ml.summary()
#p value is 0 and R sqrd value is  0.516
# Not significant

# Administration
AD_ml = smf.ols('Profit~Administration',data = rmv_influenced).fit()
AD_ml.summary()
#p value is 0.455 and R sqrd value is 0.012
#which is not significant


# Models with Transformations:

# Model 1
import numpy as np

# Model 1
# log to Profit
model1 = smf.ols('np.log(Profit)~R_and_D_Spend+Administration+Marketing_Spend',data = rmv_influenced).fit()  
model1.summary() #0.925, AIC:-84.90

# Model 2
# log to Administration
model2 = smf.ols('Profit~R_and_D_Spend+ np.log(Administration)+Marketing_Spend',data = rmv_influenced).fit()  
model2.summary() #0.962, AIC:994.9

# Model 3 
# log to Marketing_Spend
model3 = smf.ols('Profit~R_and_D_Spend+np.reciprocal(Administration)+Marketing_Spend',data = rmv_influenced).fit()  
model3.summary() # 0.962, AIC:995.4

# Model 4
rmv_influenced['Administration'] = rmv_influenced.Administration * rmv_influenced.Administration
model4 = smf.ols('Profit~R_and_D_Spend+Administration+Marketing_Spend',data = rmv_influenced).fit()  
model4.summary() # 0.963, AIC:995.4

# Prediction for final model
profit_pred_model4 = model4.predict(rmv_influenced)

# RMSE
# Error
Actual = rmv_influenced['Profit']
Prediction_model4 = profit_pred_model4

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_model4 = sqrt(mean_squared_error(Actual, Prediction_model4))
print(rmse_model4) #7028.168934502867

# Partial Regress Plot for Model4
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model4)




# preparing model with 2 inputs, signoficant will be choosen
# Except R_and_D_Spend
mlR=smf.ols('Profit~Administration+Marketing_Spend',data = rmv_influenced).fit()  
mlR.summary() # Not Significant

# Prediction 
profit_pred_mlR = mlR.predict(rmv_influenced)

Error = Actual-Prediction_mlR

# RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_mlR = sqrt(mean_squared_error(Actual, Prediction_mlR))
print(rmse_mlR) #24088.758974632925



# Except Administration
mlA=smf.ols('Profit~R_and_D_Spend+Marketing_Spend',data = rmv_influenced).fit()  
mlA.summary() #Not Significant

# Prediction
profit_pred_mlA = mlA.predict(rmv_influenced)

# Error
Actual = rmv_influenced['Profit']
Prediction_mlA = profit_pred_mlA

Error = Actual-Prediction_mlA

# RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_mlA = sqrt(mean_squared_error(Actual, Prediction_mlA))
print(rmse_mlA) #7200.9044424190015


# Except Marketing_Spend
mlM=smf.ols('Profit~Administration+R_and_D_Spend',data = rmv_influenced).fit()  
mlM.summary() # Significant

# Prediction
profit_pred_mlM = mlM.predict(rmv_influenced)

# Error
Actual = rmv_influenced['Profit']
Prediction_mlM = profit_pred_mlM

Error = Actual-Prediction_mlM

# RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_mlM = sqrt(mean_squared_error(Actual, Prediction_mlM))
print(rmse_mlM) #7160.04961495039

# the model without Marketing_Spend variable is...
# giving a Good r-sqrd value and lesser p-value
# RMSE value of the model is also lowest compared with other 2 models

# Partial Regressor Plot
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(mlM)
# creating models based on individual inputs




import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(m3_v)

# VIF for Market Spend
rsq_MS = smf.ols('Marketing_Spend~R_and_D_Spend+Administration',data = rmv_influenced).fit().rsquared  
vif_MS = 1/(1-rsq_MS)
vif_MS # 2.22986

# VIF for Adm
rsq_ADM = smf.ols('Administration~R_and_D_Spend+Marketing_Spend',data = rmv_influenced).fit().rsquared  
vif_ADM = 1/(1-rsq_ADM)
vif_ADM #1.19601

# VIF for RND
rsq_RND = smf.ols('R_and_D_Spend~Administration+Marketing_Spend',data = rmv_influenced).fit().rsquared  
vif_RND = 1/(1-rsq_RND)
vif_RND # 2.250971


 # Storing vif values in a data frame
d1 = {'Variables':['Marketing_Spend','R_and_D_Spend','Administration'],'VIF':[vif_MS,vif_RND,vif_ADM]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As RND is having higher VIF value, we are not going to include this prediction model






#...For EXTRA Knowledge..
#...similar to Dropping a Column....
#... we can add a NEW Column to data set....
State = (startup["State"])
df = pd.DataFrame(startup_new)
df['State'] = State
df