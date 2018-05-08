import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data = np.loadtxt('d1.txt')
X = data[:,:2]
Y = data[:,-1]

positiveIndices = np.where(Y == 1)[0]
negativeIndices = np.where(Y == -1)[0]

plt.plot(X[positiveIndices,0], X[positiveIndices,1], 'r.')
plt.plot(X[negativeIndices,0], X[negativeIndices,1], 'b.')
plt.show()
