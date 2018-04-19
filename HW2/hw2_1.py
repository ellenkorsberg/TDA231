import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
from sklearn import datasets



data = genfromtxt('dataset2.txt', delimiter=',')
labels = data[:,-1]


def sge(X):
    # Calculate mu
    rows = X.shape[0]
    mu = [0] * X.shape[1]
    for i in range(rows):
        mu += X[i][:]/rows

    # Calculate sigma
    sigmaSquare = 0
    for i in range(rows):
        sigmaSquare += np.dot(np.transpose(X[i][:]-mu), X[i][:]-mu)/(rows*X.shape[1])

    return mu,sigmaSquare


def sph_bayes(Xtest, trainingData, estPos1, estNeg1):
    ccdPos1 = multivariate_normal.pdf(Xtest, mean=estPos1[0], cov=[[estPos1[1], 0, 0],[0, estPos1[1], 0],[0, 0, estPos1[1]]])
    ccdNeg1 = multivariate_normal.pdf(Xtest, mean=estNeg1[0], cov=[[estNeg1[1], 0, 0],[0, estNeg1[1], 0],[0, 0, estNeg1[1]]])

    postPos1 = ccdPos1/(ccdPos1 + ccdNeg1)
    postNeg1 = ccdNeg1/(ccdPos1 + ccdNeg1)

    if postPos1 >= postNeg1:
        return [postPos1, postNeg1, 1]

    return [postPos1, postNeg1, -1]


def new_classifier(Xtest, mu1, mu2):
    b = mu1 + mu2
    b = [(1/2)*x for x in b]
    denominator = np.sqrt(np.dot(np.transpose(mu1-mu2),mu1-mu2))
    Ytest = np.sign(np.dot(np.transpose(mu1-mu2), Xtest - b)/denominator)

    return Ytest


def scaleData(data):
   scaledDataVar = []
   for i in range(len(data)):
       sample = data[i,:64]
       scaledSample = sample/255
       scaledSample = scaledSample.reshape((8,8))
       variance = np.r_[np.var(scaledSample, axis=1), np.var(scaledSample, axis=0)]
       scaledDataVar.append(np.append(variance, data[i][64]))

   return np.array(scaledDataVar)


def crossValidationBayes(data, labelIndex):
    X = data
    kf = KFold(n_splits=5, shuffle=True)
    error = 0
    for train_index, test_index in kf.split(X):
       X_train, X_test = X[train_index], X[test_index]
       estPos1 = sge(X_train[X_train[:,3] == 1][:,:3])
       estNeg1 = sge(X_train[X_train[:,3] == -1][:,:3])
       for i in range(X_test.shape[0]):
           pred = sph_bayes(X_test[:,:3][i], X_train, estPos1, estNeg1)
           if pred[2] != X_test[:][i][labelIndex]:
               error += 1

    return error/5


def crossValidationNewClassifier(data, labelIndex):
    X = data
    kf = KFold(n_splits=5, shuffle=True)
    error = 0
    for train_index, test_index in kf.split(X):
       X_train, X_test = X[train_index], X[test_index]
       mu1 = sge(X_train[X_train[:,labelIndex] == 1][:,:labelIndex])[0]
       mu2 = sge(X_train[X_train[:,labelIndex] == -1][:,:labelIndex])[0]
       for i in range(X_test.shape[0]):
           pred = new_classifier(X_test[:,:labelIndex][i], mu1, mu2)
           if pred != X_test[:][i][labelIndex]:
               error += 1

    return error/5


def firstPracticalProblem():
    data = genfromtxt('dataset2.txt', delimiter=',')
    print("Bayes classifier error: ", crossValidationBayes(data, 3))
    print("New (sign) classifier error: ", crossValidationNewClassifier(data, 3))


def getFiveAndEightClasses(digits):
    fives = []
    eights = []

    for i in range(digits.target.shape[0]):
        if digits.target[i] == 5:
            fives.append(np.append(digits.data[:][i], 1))
        elif digits.target[i] == 8:
            eights.append(np.append(digits.data[:][i], -1))

    return np.concatenate((fives,eights))


def secondPracticalProblem():
    digits = datasets.load_digits()
    data = getFiveAndEightClasses(digits)
    print("New (sign) classifier error with original feature vector: ", crossValidationNewClassifier(data, 64))
    print("New (sign) classifier error with another feature vector: ", crossValidationNewClassifier(scaleData(data), 16))

    # IS THIS NEEDED?

    # mu1 = sge(data[data[:,64] == 1][:,:64])[0] # mean of fives class
    # mu2 = sge(data[data[:,64] == -1][:,:64])[0] # mean of eights class
    # pred = new_classifier(digits.images[5].flatten(), mu1, mu2)
    # print(pred)
    # plt.matshow(digits.images[5])
    # plt.show()



#firstPracticalProblem()
secondPracticalProblem()
