###########################################################################################
#  Burak Koryan | burak@koryan.ca | http://koryan.ca
#  Project : Classification of different brands of cereals using k-NN classifier in Python
#  Date : May 5 2018
###########################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
from sklearn import preprocessing

n_neighbors = 20
dataset = pd.read_csv('cereal.csv')
X = dataset.iloc[:, [3,5]].values
y = dataset.iloc[:, 1].values
y_set = preprocessing.LabelEncoder()
y_fit = y_set.fit(y)
y_trans = y_set.transform(y)

# Count how many 'K','G','P' are there in the data array
# in order to make new arrays for data and label.
j = 0
for i in range (0,77):
    if y[i] == 'K' or y[i] == 'G' or y[i] == 'P':
        j = j+1
        
new_data = np.zeros((j,2))
new_let = [0] * j
j = 0

for i in range (0,77):
    if y[i] == 'K' or y[i] == 'G' or y[i] == 'P':
        new_data[j] = X[i]
        new_let[j] = y[i]
        j = j+1
    
h = .02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Cylassifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y_trans)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y_trans, cmap=cmap_bold,
                edgecolor='k', s=10)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()


