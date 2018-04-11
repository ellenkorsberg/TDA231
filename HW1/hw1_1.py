import numpy as np
import matplotlib.pyplot as plt

def sge(X):
    # Calculate mu
    rows = X.shape[0]
    mu = [0] * X.shape[1]
    for i in range(rows):
        mu += X[i][:]/rows

    # Calculate sigma
    sigma = 0
    for i in range(rows):
        sigma += np.dot(np.transpose(X[i][:]-mu), X[i][:]-mu)/rows
    sigma = np.sqrt(sigma)

    return mu,sigma

def myplot1(X):
    est = sge(X)
    fig, ax = plt.subplots(1)
    ax.scatter(X[:,0],X[:,1])

    theta = np.linspace(0, 2*np.pi, 100)
    for i in range(1,4):
        r = i*est[1]
        x1 = r*np.cos(theta) + est[0][0]
        x2 = r*np.sin(theta) + est[0][1]
        ax.plot(x1, x2)
        ax.set_aspect(1)

    outsideCircle = [0] * 3
    for i in range(X.shape[0]):
        xPosition = X[i][0]
        yPosition = X[i][1]
        distance = np.sqrt((xPosition-est[0][0])**2 + (yPosition-est[0][1])**2)
        if distance < est[1]:
            outsideCircle[0] += 1
        if distance < 2*est[1]:
            outsideCircle[1] += 1
        if distance < 3*est[1]:
            outsideCircle[2] += 1

    print(outsideCircle)
    ax.legend([round(outsideCircle[0]/X.shape[0],3), round(outsideCircle[1]/X.shape[0],3), round(outsideCircle[2]/X.shape[0],3)])



    plt.show()


# Load the data.
X = np.loadtxt("dataset0.txt")
X = X[:,:2]
myplot1(X)
