import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from adaline import AdalineGD

def plot_decision_regin(X, y, classifier, resolution=0.01):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1

    x2_min, x2_max = X[:,1].min()-1, X[:1].max()+1

    # 1*n, 1*m
    # meshgrid(1*n, 1*m) => 1*n=>xx1 m*n 复制m行变成m*n, 1*m=>xx2 m*n 行变列m*1，复制n列变成m*n
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min,x2_max,resolution))

    # ravel()浅复制，降维1*(mn)
    # array 变成 2*(mn)
    # 转置 mn*2,变成一个样本集，共mn个样本，每个样本两个特征
    # 得到预测结果集mn*1
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 把结果集拆分成m*n矩阵，作用是下标对应于xx1, xx2全排列的下标，格式化数据方便绘制等高图
    Z = Z.reshape(xx1.shape)

    # 把区域平分取点，构造样本集合做预测，可视化区域划分结果
    plt.contourf(xx1,xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 使用enumerate对集合加No.，
    for idx, cl in enumerate(np.unique(y)):
        # 绘制iris数据集
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx),marker=markers[idx],label=cl)


df = pd.read_csv('iris.csv',header=None)

y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa', -1, 1)

X = df.iloc[0:100, [0,2]].values

X_std = np.copy(X)
X_std[:,0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
X_std[:,1] = (X[:, 1] - X[:, 1].mean()) / X[:,1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
# 使用样本集训练模型
ada.fit(X_std, y)

# 训练完毕，根据样本集特征的最大最小值，均分特征区间构造测试样本集，测试并可视化测试结果。
plot_decision_regin(X_std, y, classifier=ada)
plt.title('Adaline - Grandient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized')
plt.legend(loc='upper left')
plt.show()
# plot()，绘制训练过程每次迭代的损失函数值。（可视化损失函数的收敛过程）
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()