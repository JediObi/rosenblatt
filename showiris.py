import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

df = pd.read_csv("iris.csv", header=None)

y = df.iloc[0:100, 4].values
y = np.where(y=="Iris-setosa", -1, 1)

X = df.iloc[0:100, [0, 2]].values

def plot_decision_region(X, y, resolution=0.01):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    # 对y去重只剩两个值-1和1,长度2，然后取出colors[0:2] 即red blue，做成colormap
    cmap = ListedColormap(colors[:len(np.unique(y))])


    # 对标签集去重，得到两个值
    # x=X[y==cl, 0]一个切片表达式，y==cl是取出y==cl的坐标范围，由于X和y是等长的一一对应的
    # 所以就相当于取出同一品种的指定特征
    # 然后指定透明度，使用enumerate获取编号用于取色，最后是对应的点图和标签
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

# plot_decision_region(X,y)
plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
plt.scatter(X[50:100, 0], X[50:, 1], color="blue", marker="x", label="versicolor")
plt.xlabel("petal length")
plt.ylabel("sepal length")
plt.legend(loc="upper left")
plt.show()