import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from matplotlib.colors import ListedColormap

df = pd.read_csv("iris.csv", header=None)

y = df.iloc[0:100, 4].values
y = np.where(y=="Iris-setosa", -1, 1)

X = df.iloc[:100, [0, 2]].values

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

# 模型学习完毕，构造离散的样本集，可视化学习结果
# 获取每个特征的极大和极小值，限定步长，在每个特征空间内生成的特征集，构造两个特征集的直积，生成一个模拟样本集
# 对模拟样本集做预测，得到模拟样本的结果集。模拟样本集分类结束
# 可视化，根据分类结果构造等高线和等高区域，使用纯色填充等高区域，边界线反映了通过特征学习得到的权重集合，即模型
def plot_decision_region(X, y, classifier, resolution=0.01):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 把两个特征的最大和最小取出来，并且加减1以增大坐标系绘制区，使离散点全都能被绘制出来
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 在两个特征的区间内，根据分辨率产生坐标轴上需要标出数值的节点。
    # 假设range1=1xm range2=1xn
    # 则 xx1 = nxm  ,  xx2 = nxm，所以xx2的列是range2转置的结果
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # A = mn
    # 使用ravel降维成向量（深拷贝）,变成了mxn共A个元素 降维成 Ax1
    # 然后把两个向量合并，构成新的矩阵 2xA
    # 则转置后 Ax2,相当于A个样本的集合
    # 然后把矩阵带入向量机，预测结果
    # 预测中使用 dot(X, w[1:])+w[0]获取激活值,Ax2 dot 2x1 => Ax1
    # 由于python的矩阵+常量=每个元素加常量，所以构成一个激活值矩阵 Ax1
    # 对激活值矩阵运用where就生成了预测结果集 Ax1
    Z = classifier.predict(np.array([xx1.ravel(),  xx2.ravel()]).T)
    # 对预测结果集reshape（实际上,前边的步骤已经使结果集和xx1都变成了Ax1的shape）
    Z = Z.reshape(xx1.shape)

    # contourf()等高线工具，这个工具不绘制等高线，但是会使用颜色填充等高区域
    # 等高线的区域等效于contour()，声明的坐标绘制等高线，未声明的坐标，按所在区间填充颜色
    # 区域 x = Ax1, y= Ax1, 前二者是坐标区域，Z = A x 1表示高度数据，比如x和y里各两个数据，相当于两个点，那么这中间的连线就会被标注成Z里某个值
    # cmap指定了不同区域对应的颜色
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # 限制绘制区域
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 对原始数据绘制散点图
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
plot_decision_region(X, y, classifier=ppn)
plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc="upper left")
plt.show()