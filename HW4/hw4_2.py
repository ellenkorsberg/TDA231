import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from itertools import product

data = np.loadtxt('d2.txt')
X = data[:,:2]
Y = data[:,-1]

positiveIndices = np.where(Y == 1)[0]
negativeIndices = np.where(Y == -1)[0]

clfPoly = SVC(kernel='poly', degree=2)
clfRbf = SVC(kernel='rbf')
clfLinear = SVC(kernel='linear')
clfPoly.fit(X, Y)
clfRbf.fit(X, Y)
clfLinear.fit(X, Y)

plt.figure(0)
plt.title('SVM with linear kernel')
# Plotting the data.
plt.plot(X[positiveIndices,0], X[positiveIndices,1], 'ro', markersize=8, markeredgecolor='none')
plt.plot(X[negativeIndices,0], X[negativeIndices,1], 'bo', markersize=8, markeredgecolor='none')

# Marking misclassified points.
misclassified = 0
for i in range(1,200):
    if clfLinear.predict([[X[i,0], X[i,1]]]) != Y[i]:
        misclassified += 1
        plt.plot(X[i,0], X[i,1], 'go', markersize=5, markeredgecolor='none')


# Plotting decision regions for the nonlinear kernels.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

plt.figure(1)
plt.title('SVM with quadratic kernel')
for idx, clfPoly, tt in zip(product([0, 1], [0, 1]), [clfPoly], ['Quadratic kernel']):
    Z = clfPoly.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=Y, marker='o', s=50, edgecolors='none')

plt.figure(2)
plt.title('SVM with rbf kernel')
for idx, clfRbf, tt in zip(product([0, 1], [0, 1]), [clfRbf], ['Rbf kernel']):
    Z = clfRbf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=Y, marker='o', s=50, edgecolors='none')

plt.figure(0)
plt.show()
