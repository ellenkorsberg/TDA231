import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold



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



def new_classifier(Xtest, mu1, mu2):
    b = mu1 + mu2
    b = [(1/2)*x for x in b]
    denominator = np.sqrt(np.dot(np.transpose(mu1-mu2),mu1-mu2))
    Ytest = np.sign(np.dot(np.transpose(mu1-mu2), Xtest - b)/denominator)

    return Ytest



#
# # 5-fold cross validation error for Bayes classifier
# X = data
# kf = KFold(n_splits=5, shuffle=True)
# error = 0
# for train_index, test_index in kf.split(X):
#    X_train, X_test = X[train_index], X[test_index]
#    for i in range(X_test.shape[0]):
#        pred = sph_bayes(X_test[:,:3][i], X_train)
#        if pred[2] != X_test[:][i][3]:
#            error += 1
#
# print("Bayes error: ", error/5)
#
# # 5-fold cross validation error for other classifier
# error = 0
# for train_index, test_index in kf.split(X):
#    X_train, X_test = X[train_index], X[test_index]
#    for i in range(X_test.shape[0]):
#        mu1 = sge(X_train[X_train[:,3] == 1][:,:3])[0]
#        mu2 = sge(X_train[X_train[:,3] == -1][:,:3])[0]
#        pred = new_classifier(X_test[:,:3][i], mu1, mu2)
#        if pred != X_test[:][i][3]:
#            error += 1
#
# print("Other error: ", error/5)



import numpy as np
from sklearn import datasets
digits = datasets.load_digits()

data = digits.data
target_names = digits.target_names
import matplotlib.pyplot as plt
y = digits.target


fives = []
eights = []

for i in range(digits.target.shape[0]):
    if digits.target[i] == 5:
        fives.append(digits.data[:][i])
    elif digits.target[i] == 8:
        eights.append(digits.data[:][i])

fives = np.array(fives)
eights = np.array(eights)


mu1 = sge(fives)[0]
mu2 = sge(eights)[0]
pred = new_classifier(digits.images[5].flatten(), mu1, mu2)
print(pred)
plt.matshow(digits.images[5])
#plt.show()

scaledDataVar = []
for i in range(len(data)):
    sample = data[i][:]
    maxGrey = np.max(sample)
    scaledSample = sample/maxGrey
    scaledSample = scaledSample.reshape((8,8))
    variance = np.zeros(16)
    # rows:
    for j in range(len(scaledSample)): #len(scaledSample)
        variance[j] = np.var(scaledSample[j][:])
        variance[j+8] = np.var(scaledSample[:][j])

    scaledDataVar.append(variance)

print(np.array(scaledDataVar).shape)
