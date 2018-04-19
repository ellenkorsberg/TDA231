import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

data = genfromtxt('dataset2.txt', delimiter=',')
labels = data[:,-1]


def sge(X):
    # Calculate mu
    rows = X.shape[0]
    mu = [0] * X.shape[1]
    for i in range(rows):
        mu += X[i][:]/rows

    # Calculate sigma
    sigma = 0
    for i in range(rows):
        sigma += np.dot(np.transpose(X[i][:]-mu), X[i][:]-mu)/(rows*X.shape[1])
    #sigma = np.sqrt(sigma)

    return mu,sigma


def sph_bayes(Xtest, trainingData): # other parameters needed.
    estPos1 = sge(trainingData[trainingData[:,3] == 1][:,:3])
    estNeg1 = sge(trainingData[trainingData[:,3] == -1][:,:3])
    ccdPos1 = multivariate_normal.pdf(Xtest, mean=estPos1[0], cov=[[estPos1[1], 0, 0],[0, estPos1[1], 0],[0, 0, estPos1[1]]])
    ccdNeg1 = multivariate_normal.pdf(Xtest, mean=estNeg1[0], cov=[[estNeg1[1], 0, 0],[0, estNeg1[1], 0],[0, 0, estNeg1[1]]])

    postPos1 = ccdPos1/(ccdPos1 + ccdNeg1)
    postNeg1 = ccdNeg1/(ccdPos1 + ccdNeg1)

    if postPos1 >= postNeg1:
        return [postPos1, postNeg1, 1]

    return [postPos1, postNeg1, -1]


#print(sph_bayes(data[:,:3][1500], data))

def new_classifier(Xtest, mu1, mu2):
    b = mu1 + mu2
    b = [(1/2)*x for x in b]
    print(b)
    denominator = np.sqrt(np.dot(np.transpose(mu1-mu2),mu1-mu2))
    print(denominator)
    Ytest = np.sign(np.dot(np.transpose(mu1-mu2), Xtest - b)/denominator)

    return [Ytest]

mu1 = sge(data[data[:,3] == 1][:,:3])
mu1 = mu1[0]
mu2 = sge(data[data[:,3] == -1][:,:3])
mu2 = mu2[0]
print(new_classifier(data[1500,:3], mu1, mu2))