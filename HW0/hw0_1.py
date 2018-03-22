import numpy as np
import matplotlib.pyplot as plt
import operator
import sys

# Predefinitions
mean = [1, 1]
cov = [[0.1, -0.05], [-0.05, 0.2]]

# Define the function f.
def f(x,r):
    return (np.array(list(map(operator.sub, x, mean))).dot(np.linalg.inv(cov).dot(np.transpose(list(map(operator.sub, x, mean))))))/2 - r

# Plot level sets f(x,r)=0 for r=1,2,3.
x = np.linspace(-1, 3, num=200)
y = np.linspace(-1, 3, num=200)
xx,yy = np.meshgrid(x, y)
levels = []

for r in range(1,4):
    levels.append([])
    for i in range(len(x)):
        levels[r-1].append([])
        for j in range(len(y)):
            xy = [x[i], y[j]]
            levels[r-1][i].append(f(xy,r))

array1 = np.array(levels[0])
array2 = np.array(levels[1])
array3 = np.array(levels[2])
plt.contour(xx, yy, array1, [0], colors = 'r')
plt.contour(xx, yy, array2, [0], colors = 'r')
plt.contour(xx, yy, array3, [0], colors = 'r')

# Gather and plot points outside and inside the level set f(x,3)=0.
points = np.random.multivariate_normal(mean, cov, 1000)
bluePoints = []
blackPoints = []
for i in range(len(points)):
    if f(points[i], 3) > 0:
        blackPoints.append(points[i])
    else:
        bluePoints.append(points[i])

plt.plot(np.transpose(blackPoints)[1], np.transpose(blackPoints)[0], 'k.')
plt.plot(np.transpose(bluePoints)[1], np.transpose(bluePoints)[0], 'b.')

# Set title of plot to number of points outside level set f(x,3)=0.
plt.title('Number of black points: ', loc='left')
plt.title(len(blackPoints))

plt.show()
