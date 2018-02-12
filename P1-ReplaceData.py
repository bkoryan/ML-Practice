# B.KORYAN - Feb 11 2018
# web: http://koryan.ca | e-mail: burak@koryan.ca
# Machine Learning practice in Python
# Description : Replacing missing numerical data by
# taking average of column
#########################################################
# Dataset (data1.csv)
#   A   0.1
#   B   3.0
#   C   4.0
#   D   NaN
#########################################################


# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data1.csv')  # Import the dataset
X = dataset.iloc[:,:2].values       # take rows data into variable X

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(X[:,1:3])         # columns 1 and 2 selected
X[:,1:3] = imputer.transform(X[:,1:3])  # transform dataset
print(X)                                # print X after replacing missing data
