from sklearn import datasets
digits = datasets.load_digits()

data = digits.data
print(data.shape)
target_names = digits.target_names
print (target_names)
import matplotlib.pyplot as plt
y = digits.target
plt.matshow(digits.images[5])
plt.show()
