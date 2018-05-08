import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import NuSVC

data = np.loadtxt('d1.txt')
X = data[:,:2]
Y = data[:,-1]

positiveIndices = np.where(Y == 1)[0]
negativeIndices = np.where(Y == -1)[0]

clf = SVC(kernel='linear')
clf.fit(X, Y)

# Plotting the data and the decision boundary
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(np.min(X)-1, np.max(X)+1)
yy = a * xx - (clf.intercept_[0]) / w[1]
plt.plot(X[positiveIndices,0], X[positiveIndices,1], 'r.')
plt.plot(X[negativeIndices,0], X[negativeIndices,1], 'b.')
plt.plot(xx, yy, 'k-')
plt.show()
