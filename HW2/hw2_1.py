from numpy import genfromtxt
data = genfromtxt('dataset2.txt', delimiter=',')
labels = data[:,-1]
