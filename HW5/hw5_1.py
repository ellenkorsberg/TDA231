import scipy.io
import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy

mat = scipy.io.loadmat('hw5_p1b.mat')
X = mat['X']

def kMeans(X, k):
    # Initialize centroids
    mu = []
    for i in range(k):
        mu.append(X[np.random.randint(low = 0, high = 250),:])

    Z = np.zeros((X.shape[0], len(mu)))
    dist = np.zeros((X.shape[0], len(mu)))
    iterations = 0
    assignmentAfter2Iter = []

    while True:

        if iterations == 2:
            assignmentAfter2Iter = deepcopy(Z)

        # Copy latest Z in order to decide if assingments changed
        formerZ = deepcopy(Z)

        # Assign every point to its closest centroid
        for i in range(X.shape[0]):
            for j in range(len(mu)):
                dist[i,j] = np.sqrt((X[i,0] - mu[j][0])**2 + (X[i,1] - mu[j][1])**2)
                Z[i,j] = int( np.argmin(dist[i]) == j )

        # Update centroids
        for i in range(len(mu)):
            sum1 = 0
            for j in range(X.shape[0]):
                sum1 += Z[j][i] * X[j,:]
            mu[i] = sum1/( np.sum(Z[:,i], axis = 0) )

        # Check if assignments have changed
        if np.array_equal(formerZ, Z):
            break

        iterations += 1

    # Mark out which data points that changed assingment since the second iteration
    changedAssignments = X[np.unique(np.where((assignmentAfter2Iter == Z) == False)[0])]
    plt.scatter(changedAssignments[:,0], changedAssignments[:,1], marker = "o", color = "y", s = 40, alpha = 0.7)

    print("Iterations: %s" % iterations)
    colors = itertools.cycle(["r", "b", "g", "m", "y"])
    for i in range(len(mu)):
        classColor = next(colors)
        plt.scatter(mu[i][0], mu[i][1], color = classColor, marker = "o", s = 70)
        classData = X[np.where(np.where(Z == 1)[1] == i)]
        plt.scatter(classData[:,0], classData[:,1], color = classColor, alpha = 0.8, marker = ".")

    plt.show()
    return

def gaussianRBFkernel(x, y):
    sigma = 0.2
    norm = (np.linalg.norm(x-y))**2
    return math.exp(-norm/(2*sigma**2))

def calculateDistance(X, k, z, index):
    N = np.sum(z[:,k])

    sum1 = 0
    sum2 = 0
    for i in range(X.shape[0]):
        sum1 = sum1 + z[i,k]*gaussianRBFkernel(X[index,:], X[i,:])
        for j in range(X.shape[0]):
            sum2 = sum2 + z[i,k]*z[j,k]*gaussianRBFkernel(X[i,:], X[j,:])

    return gaussianRBFkernel(X[index,:], X[index,:]) - 2*(1/N)*sum1 + (1/N**2)*sum2


def kernelKMeans(X, k):

    Z = np.zeros((X.shape[0], k))

    for i in range(X.shape[0]):
        rand = np.random.randint(low = 0, high = X.shape[1])
        Z[i,rand] = 1


    formerZ = []
    dist = np.zeros((X.shape[0], k))
    iterations = 0

    while True:

        formerZ = deepcopy(Z)
        for i in range(X.shape[0]):
            for j in range(k):
                dist[i,j] = calculateDistance(X, j, Z, i)
            for j in range(k):
                Z[i,j] = int(np.argmin(dist[i]) == j)

        if np.array_equal(formerZ, Z):
            break

        iterations += 1

    print("Iterations: %s" % iterations)
    colors = itertools.cycle(["r", "b", "g", "m", "y"])
    for i in range(k):
        classColor = next(colors)
        classData = X[np.where(np.where(Z == 1)[1] == i)]
        plt.scatter(classData[:,0], classData[:,1], color = classColor, alpha = 0.8, marker = ".")

    plt.show()

kernelKMeans(X, 2)
