import numpy as np
import matplotlib.pyplot as plt
import operator
import sys

mean = [1, 1]
cov = [[0.1, -0.05], [-0.05, 0.2]]


def f(x,r):
    inverse = np.linalg.inv(cov)
    diff = list(map(operator.sub, x, mean))
    return (np.transpose(diff).dot(inverse).dot(diff))/2 - r
    #return ((np.array(map(operator.sub,x,mean)).T.tolist()*np.linalg.inv(cov)*map(operator.sub,x,mean))/2) - r

x_ = np.linspace(-5, -4.5, num=200)
y_ = np.linspace(-5, -4.5, num=200)
x,y = np.meshgrid(x_, y_)
levels = []

for r in range(1,4):
    levels.append([])
    for i in range(len(x)):
        levels[r-1].append([])
        for j in range(len(y)):
            levels[r-1][i].append(f([i,j],r))

print(np.array(levels).shape)

array1 = np.array(levels[0])
array2 = np.array(levels[1])
array3 = np.array(levels[2])

c1 = plt.contour(x, y, array1, [0])
c2 = plt.contour(x, y, array2, [0])
c2 = plt.contour(x, y, array3, [0])
plt.show()
sys.exit()


#x = np.random.multivariate_normal(mean, cov, 5).T
#res = f(x,1)
#print res


# You can use, np.meshgrid() and np.contour to make your life easier
