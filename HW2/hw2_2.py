import numpy as np
from sklearn import datasets
digits = datasets.load_digits()

data = digits.data
target_names = digits.target_names
import matplotlib.pyplot as plt
y = digits.target
<<<<<<< HEAD
#plt.matshow(digits.images[5])
#plt.show()

print(digits.data)
print(digits.target)

fives = []
eights = []

for i in range(digits.target.shape[0]):
    if digits.target[i] == 5:
        fives.append(digits.data[:][i])
    elif digits.target[i] == 8:
        eights.append(digits.data[:][i])

print(np.array(fives).shape)
print(np.array(eights).shape)
=======
plt.matshow(digits.images[5])
plt.show()
>>>>>>> HW2
