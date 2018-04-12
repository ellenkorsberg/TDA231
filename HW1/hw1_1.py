import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import invgamma

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
    sigma = np.sqrt(sigma)

    return mu,sigma

def myplot1(X):
    est = sge(X)
    mu = est[0]
    sigma = est[1]

    fig, ax = plt.subplots(1)
    # Scatter plot the points
    ax.scatter(X[:,0],X[:,1],s=5)

    # Plot circles
    theta = np.linspace(0, 2*np.pi, 100)
    for i in range(1,4):
        r = i*sigma
        x1 = r*np.cos(theta) + mu[0]
        x2 = r*np.sin(theta) + mu[1]
        ax.plot(x1, x2)
        ax.set_aspect(1)

    # Collect legend information
    outsideCircle = [0] * 3
    nbrOfPoints = X.shape[0]
    for i in range(nbrOfPoints):
        xPosition = X[i][0]
        yPosition = X[i][1]
        distance = np.sqrt((xPosition-mu[0])**2 + (yPosition-mu[1])**2)
        if distance < sigma:
            outsideCircle[0] += 1
        if distance < 2*sigma:
            outsideCircle[1] += 1
        if distance < 3*sigma:
            outsideCircle[2] += 1

    ax.legend([round(outsideCircle[0]/nbrOfPoints,3), round(outsideCircle[1]/nbrOfPoints,3), round(outsideCircle[2]/nbrOfPoints,3)])
    plt.show()


# Load the data.
X = np.loadtxt("dataset0.txt")
# Extract the first two features
X = X[:,:2]
print(sge(X))



def myplot2(X):
    fig, ax1 = plt.subplots(1)
    mu = sge(X)[0]
    # Plot the prior and posterior
    s = np.sum(np.einsum('ij,ij->i',X-mu,X-mu))/2
    sigma = np.linspace(0.000001, 200, num=500)

    alphaPrior1 = 1
    alphaPost1 = 1+(X.shape[1]*X.shape[0]/2)
    alphaPrior2 = 10
    alphaPost2 = 10+(X.shape[1]*X.shape[0]/2)
    beta = 1 + s

    plt.subplot(2, 1, 1)
    plt.plot(sigma, invgamma.pdf(sigma**2, alphaPrior1))
    plt.plot(sigma, invgamma.pdf(sigma**2, alphaPost1, scale = beta))
    plt.title('alpha = 1')
    plt.legend(['Prior', 'Posterior'])

    plt.subplot(2, 1, 2)
    plt.plot(sigma, invgamma.pdf(sigma**2, alphaPrior2))
    plt.plot(sigma, invgamma.pdf(sigma**2, alphaPost2, scale = beta))
    plt.title('alpha = 10')
    plt.legend(['Prior', 'Posterior'])

    plt.show()

myplot2(X)
