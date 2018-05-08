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

# Plotting the data and the decision boundary.
plt.plot(X[positiveIndices,0], X[positiveIndices,1], 'ro')
plt.plot(X[negativeIndices,0], X[negativeIndices,1], 'bo')
plt.plot(X[clf.support_,0], X[clf.support_, 1], 'w.')

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(np.min(X)-1, np.max(X)+1)
yy = a * xx - (clf.intercept_[0]) / w[1]
yyPos1 = 1/w[1] + a * xx - (clf.intercept_[0]) / w[1]
yyNeg1 = -1/w[1] + a * xx - (clf.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k-')
plt.plot(xx, yyPos1, 'm--')
plt.plot(xx, yyNeg1, 'm--')

# Marking misclassified points.
misclassified = 0
for i in range(1,200):
    if clf.predict([[X[i,0], X[i,1]]]) != Y[i]:
        misclassified += 1
        plt.plot(X[i,0], X[i,1], 'g.')

# Calculating and printing the (soft) margin.
margin = 2/np.sqrt(w[0]**2 + w[1]**2)
print("The margin is %s" % margin)

# Printing the bias.
print("The bias is %s" % clf.intercept_[0])

plt.show()
