#############################################################
# Burak Koryan | burak@koryan.ca | http://koryan.ca          
# Date : May 1 2018
# Project: Alcohol percentage prediction of Beer type using 
#          simple Linear Regression in Python
############################################################

# Definition of Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datashet and sorting data   						
dataset = pd.read_csv('beers.csv')
X = dataset.iloc[1:40,5].values                                          	# Select beer type column
Y = dataset.iloc[1:40,1].values                                           	# Select alcohol percentage column
print(Y)                                                             	     	# Check X & Y variables
print(X)

# Encoding data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder			# Import encoder library		 	
labelencoder_X = LabelEncoder()							
X[:] = labelencoder_X.fit_transform(X[:])					# Encode the beer-type columm X
onehotencoder = OneHotEncoder(categorical_features=[0])				

# Cross validation and data splitting
from sklearn.cross_validation import train_test_split				# Import cross val library from sklearn
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.02,random_state = 0)		# test size = 20%

# Linear regression starts: 
from sklearn.linear_model import LinearRegression				# import regression model from sklearn
X_train = X_train.reshape(-1,1)							# Reshape the X&Y Variables
y_train = y_train.reshape(-1,1)			
regressor = LinearRegression()				
regressor.fit(X_train,y_train)							# fitting training data
X_test = X_test.reshape(-1,1)				
y_pred = regressor.predict(X_test)						# predicting alcohol percentage

# Plotting data
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Beer Type v.s. Alcohol(%) prediction using Linear Regression')
plt.xlabel('Beer Type')
plt.ylabel('Alcohol %')
plt.show()
