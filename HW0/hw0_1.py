import numpy as np
import matplotlib.pyplot as plt
import operator
import sys

mean = [1, 1]
cov = [[0.1, -0.05], [-0.05, 0.2]]


def f(x,r):
    return ((np.array(map(operator.sub,x,mean)).T.tolist()*np.linalg.inv(cov)*map(operator.sub,x,mean))/2) - r

x_ = np.linspace(-5, 5, num=200)
y_ = np.linspace(-5, 5, num=200)
x,y = np.meshgrid(x_, y_)
levels = []


for i in range(len(x)):
    levels.append([])
    for j in range(len(y)):
        levels[i].append(f([i,j],1))

array = np.array(levels)
array = array[:, :, 0, 0] # remove the two unwanted dimensions


c = plt.contour(x, y, array)
plt.colorbar()
plt.show()
sys.exit()


#x = np.random.multivariate_normal(mean, cov, 5).T
#res = f(x,1)
#print res


# You can use, np.meshgrid() and np.contour to make your life easier
