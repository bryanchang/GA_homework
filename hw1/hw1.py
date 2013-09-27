from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
    
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)

np.random.seed(0)
indices = np.random.permutation(len(iris_X)) #len(iris_X) = 150
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
prediction = knn.predict(iris_X_test)

def accuracy(prediction, iris_y_test):
    diff_count = 0
    for i, j in zip(prediction, iris_y_test):
        if i != j:
            diff_count +=1
    return diff_count
print "Only " + str(accuracy(prediction, iris_y_test)) + " data point was classified incorrectly"

print "In other words, there was " + str(int((1 - accuracy(prediction, iris_y_test)/len(prediction))*100))+"%" + " accuracy"

#knn.score(iris_X_train, iris_y_train)
knn.score(iris_X_test, iris_y_test)

import pylab as pl
from matplotlib.colors import ListedColormap

n_neighbors = 15

X = iris.data[:, :2] #for visualization, we are only grabbing the first two features
Y = iris.target

h = .02  # step size in the mesh

# create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) #color for the boundaries
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) #color for the points

n_neighbors = 15

for weights in ['uniform', 'distance']:
    # create an instance of Neighbours Classifier and fit the data
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # plot the decision boundary: assign a color to each point in the mesh [x_min, m_max]x[y_min, y_max]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # put the result into a color plot
    Z = Z.reshape(xx.shape)

    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # plot also the training points
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())
    pl.title("3-Class classification (k = %i, weights = '%s')"
             % (n_neighbors, weights))
pl.show()
