#############################################################
# Burak Koryan | burak@koryan.ca | http://koryan.ca          
# Date : May 1 2018
# Project: Alcohol percentage prediction of Beer type using 
#          simple Linear Regression in Python
############################################################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('beers.csv')
X = dataset.iloc[1:40,5].values
Y = dataset.iloc[1:40,1].values
print(Y)
print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:] = labelencoder_X.fit_transform(X[:])
onehotencoder = OneHotEncoder(categorical_features=[0])

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.02,random_state = 0)

from sklearn.linear_model import LinearRegression
X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
X_test = X_test.reshape(-1,1)
y_pred = regressor.predict(X_test)

plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Beer Type v.s. Alcohol(%) prediction using Linear Regression')
plt.xlabel('Beer Type')
plt.ylabel('Alcohol %')
plt.show()
