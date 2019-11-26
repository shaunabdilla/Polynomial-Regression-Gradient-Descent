# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:25:41 2019

@author: Shaun
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seabornInstance
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[9]:


data = pd.read_csv(r"C:\Users\Shaun\Documents\MSc AI\Semester 1\Applied Machine Learning\Lecture Notes\car_insurance_data.csv")
data.head()
data.describe()


# In[13]:


data.plot(x="Kilometres", y="Market Value", style="o")
plt.title('Kilometres vs Market Value')
plt.xlabel('Kilometres')
plt.ylabel('Market Value')
plt.show()


# In[19]:

plt.figure()
plt.tight_layout()
seabornInstance.distplot(data['Market Value'])

X = data['Kilometres'].values.reshape(-1,1)
y = data['Market Value']
X_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
model = LinearRegression().fit(X_, y)
r_sq = model.score(X_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#regressor = LinearRegression()
#regressor.fit(X_train, y_train) #train the algorithm

#
#X_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
#y = data['Market Value']
#X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=0)
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
#
#print(regressor.intercept_) #Intercept
#print(regressor.coef_) #slope
#
#y_pred = regressor.predict(X_test)
#df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted':y_pred.flatten()})
#df


# In[51]:


df.plot(kind='bar', figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.scatter(X_, y, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Kilometres vs Market Value(Prediction)}')
plt.xlabel('Kilometres')
plt.ylabel('Market Value')
plt.show()
