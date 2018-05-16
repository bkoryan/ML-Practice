####################################################################################
# Burak Koryan | burak@koryan.ca | May 16 2018
# Description : k-NN Classification of Census Demographic data from four US States
#               by State,percentage of population under poverty line and percentage
#               of youth in poverty.The colors that are used to represent the States
#               in scatter plots are red,yellow,blue,and green for Alabama,Arizona,Arkansas
#               and California,respectively.
#   Course taken : Machine Learning Algorithms in Python and R on Udemy.com
####################################################################################

# Step 1 : Import libraries that need to be used
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn import svm, datasets,neighbors
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Step 2 : Import the dataset
dataset = pd.read_csv('MLdata.csv')
Y = dataset.iloc[:,1].values
X = dataset.iloc[:,[17,18]].values

# Step 3 : Label encoding for y-axis ( the variable Y has data for the four states)
y_set = preprocessing.LabelEncoder()
y_fit = y_set.fit(Y)
y_trans = y_set.transform(Y)
n_neighbors = 5                                # number of neighbors for voting
h = .02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#FFABAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#FF00B0'])


# Step 4 : Classification starts here in the for-loop
for weights in ['uniform']:            
   # k-NN Classifier setup.'weights' can be either uniform or distance and the
   # n_neighbors variable can be any number from 1 to max number of data points.

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X,y_trans)

    # Scatter plot chart setup here    
    x_min, x_max = new_data[:, 0].min() - 1, new_data[:, 0].max() + 1
    y_min, y_max = new_data[:, 1].min() - 1, new_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Scatter plot created here.   
    plt.scatter(new_data[:, 0], new_data[:, 1], c=y_trans, cmap=cmap_bold,
                edgecolor='k', s=10)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()

# Step 5 : Confusion matrix and accuracy calculation
pred = clf.predict(new_data)
cm = confusion_matrix(y_trans,pred)
print accuracy_score(pred,y_trans)
print(cm)
