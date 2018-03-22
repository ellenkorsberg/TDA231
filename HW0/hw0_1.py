import numpy as np
import matplotlib.pyplot as plt
import operator
import sys

mean = [1, 1]
cov = [[0.1, -0.05], [-0.05, 0.2]]


def f(x,r):
    inverse = np.linalg.inv(cov)
    diff = list(map(operator.sub, x, mean))
    return (np.transpose(diff).dot(inverse.dot(diff)))/2 - r
    #return ((np.array(map(operator.sub,x,mean)).T.tolist()*np.linalg.inv(cov)*map(operator.sub,x,mean))/2) - r

x = np.linspace(-10, 10, num=200)
y = np.linspace(-10, 10, num=200)
xx,yy = np.meshgrid(x, y)
levels = []

for r in range(1,4):
    levels.append([])
    for i in range(len(x)):
        levels[r-1].append([])
        for j in range(len(y)):
            xy = [x[i], y[j]]
            levels[r-1][i].append(f(np.transpose(xy),r))

array1 = np.array(levels[0])
array2 = np.array(levels[1])
array3 = np.array(levels[2])
'''
array1 = array1[:, :, 0, 0]
array2 = array2[:, :, 0, 0]
array3 = array3[:, :, 0, 0]
'''

c1 = plt.contour(xx, yy, array1, [0], colors = 'r')
c2 = plt.contour(xx, yy, array2, [0], colors = 'g')
c3 = plt.contour(xx, yy, array3, [0], colors = 'b')
plt.show()
sys.exit()

points = np.random.multivariate_normal(mean, cov, 10)
bluePoints = []
blackPoints = []
for i in range(len(points)):
    xy = points[i]
    if (f(xy, 3)) > 0:
        bluePoints.append(xy)
    else:
        blackPoints.append(xy)

print(bluePoints)
print(blackPoints)

#plt.plot(blackPoints)
#plt.show()
#res = f(x,1)
#print res
