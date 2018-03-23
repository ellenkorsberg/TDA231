import numpy as np
import matplotlib.pyplot as plt

# Load the data.
X = np.loadtxt("dataset0.txt")

# Construct the scaled data.
Y = []
for i in range(len(X[0])):
    Y.append([])
    maxValue = np.max(X[:,i])
    for j in range(len(X)):
        Y[i].append(X[j,i]/maxValue)

# To get the axes correct.
X = np.transpose(X)

# Identify pair of features in Y that has minimum correlation and scatter plot their data.
covY = np.cov(Y)
i,j = np.where(covY == covY.min())
feature1 = X[i[0],:]
feature2 = X[j[0],:]

plt.plot(feature1, feature2, '.')
plt.title('Feature indices: (%s, %s), correlation value: %s'%(i[0], j[0], covY[i[0],j[0]]))

# Plot all covariance and correlation matrices.
plt.matshow(np.cov(X))
plt.colorbar()
plt.title('Covariance of X')
plt.matshow(np.corrcoef(X))
plt.colorbar()
plt.title('Correlation of X')
plt.matshow(np.cov(Y))
plt.colorbar()
plt.title('Covariance of Y')
plt.matshow(np.corrcoef(Y))
plt.colorbar()
plt.title('Correlation of Y')

plt.show()
