import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from perceptron import Perceptron

df = pd.read_csv("iris.csv", header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

X = df.iloc[0:100, [0, 2]]

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker="0")
plt.xlabel("Epoches")
plt.ylabel("Number of misclassifications")
plt.show()
