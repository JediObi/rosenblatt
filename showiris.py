import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("iris.csv", header=None)

y = df.iloc[0:100, 4].values
y = np.where(y=="Iris-setosa", -1, 1)

X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
plt.scatter(X[50:100, 0], X[50:, 1], color="blue", marker="x", label="versicolor")
plt.xlabel("petal length")
plt.ylabel("sepal length")
plt.legend(loc="upper left")
plt.show()