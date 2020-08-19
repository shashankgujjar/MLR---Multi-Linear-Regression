# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:18:42 2020

@author: user
"""
import pandas as pd
computer = pd.read_csv("E:\Data\Assignments\i made\MLR\Computer_Data.csv")
computer.columns
computer.shape

# data frame creation
df = pd.DataFrame(computer, columns=['sl_no', 'price', 'speed', 'hd', 'ram', 'screen', 'cd', 'multi', 'premium', 'ads', 'trend'])
print(df.shape)

# removing sl_no column
computer_new = df.drop(['sl_no'] , axis='columns')
print(computer_new.shape)

# creationg dummies, converting yes/no to 1/0
# for cd, multi, premium columns

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

computer_new['cd'] = le.fit_transform(computer_new['cd'])
computer_new['multi'] = le.fit_transform(computer_new['multi'])
computer_new['premium'] = le.fit_transform(computer_new['premium'])


#creating pairplot to check collinearity
import seaborn as sns
sns.pairplot(computer_new)
computer_new.corr()


# Model Construction

# Model 1
import statsmodels.formula.api as smf # for regression model

model1 = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=computer_new).fit() # regression model
model1.summary() # 0.776

# influence plot
import statsmodels.api as sm
sm.graphics.influence_plot(model1)

comp_influenced = computer_new.drop(computer_new.index[[1700,5960]],axis=0)
print(comp_influenced.shape)

# Model 2 based on data removed influece data points
model2 = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=comp_influenced).fit()
model2.summary() #  0.777


# Model 3
import numpy as np

model3 = smf.ols('np.log(price)~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=comp_influenced).fit()
model3.summary() # 0.784

# Model 4

comp_influenced['speed_sq'] = comp_influenced.speed * comp_influenced.speed
comp_influenced['hd_sq'] = comp_influenced.hd * comp_influenced.hd
comp_influenced['ram_sq'] = comp_influenced.ram * comp_influenced.ram
comp_influenced['screen_sq'] = comp_influenced.screen * comp_influenced.screen
comp_influenced['ads_sq'] = comp_influenced.ads * comp_influenced.ads
comp_influenced['trend_sq'] = comp_influenced.trend * comp_influenced.trend

model4 = smf.ols('price~speed+speed_sq+hd+hd_sq+ram+ram_sq+screen+screen_sq+cd+multi+premium+ads+ads_sq+trend+trend_sq',data=comp_influenced).fit()
model4.summary() # 0.805

# Price Predictions
price_pred = model4.predict(comp_influenced)










