import matplotlib.pyplot as plt
from utils.data_path import M_SET, IRIS
from sklearn.linear_model import RidgeCV
import pandas as pd
import numpy as np


def plot_ridge_curve(x, y, plist, k_max=1, q_num=10, intercept=True):
    """
    绘制岭迹曲线
    :param x : 自变量的数据矩阵
    :param y : 响应变量向量或矩阵
    :param plist : 选择显示的系数列表
    :param k_max : 岭参数的最大值
    :param q_num : 将0~k_max的区间分成q_num等分
    :param intercept : 是否计算截距
    """
    if intercept:
        x = np.c_[x, [1] * x.shape[0]]

    coefs = []
    for k in np.linspace(0, k_max, q_num + 1):
        coefs.append(np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x) + k * np.identity(x.shape[1])), x.T), y))

    coefs = np.array(coefs)
    plt.axhline(0, 0, k_max, linestyle='--', c='gray')
    plt.axhline(np.mean(coefs[:, plist]), 0, k_max, linestyle='--', c='gray')

    for p in plist:
        plt.plot(np.linspace(0, k_max, q_num + 1), coefs[:, p], '-', label=r"$\beta_" + str(p + 1) + "(k)$",
                 color='black',
                 linewidth=p + 1)
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$\beta(k)$", fontsize=14)
    plt.legend()
    plt.show()


out = pd.read_csv(M_SET)
X = out.drop(columns='y').values
Y = out.y.values
plot_ridge_curve(X, Y, [0, 1], k_max=1, q_num=100)


def get_best_k(x, y, k_max=1, q_num=10, intercept=True):
    """
    根据GCV方法，获得最佳岭参数k 
    :param x : 自变量的数据矩阵
    :param y : 响应变量向量或矩阵
    :param k_max : 岭参数的最大值
    :param q_num : 将0~k_max的区间分成q_num等分
    :param intercept : 是否计算截距
    """
    n = x.shape[0]
    if intercept:
        x = np.c_[x, [1] * n]

    gcv_list = []
    k_values = np.linspace(0, k_max, q_num + 1)
    for k in k_values:
        mk = np.matmul(np.matmul(x, np.linalg.inv(np.matmul(x.T, x) + k * np.identity(x.shape[1]))), x.T)
        yk = np.matmul(mk, y)
        trm_k = np.trace(mk)
        gcv = np.sum((y - yk) ** 2) / (n * (1 - trm_k / n) ** 2)
        gcv_list.append(gcv)
    return k_values[np.argmin(gcv_list)], np.min(gcv_list)


iris = pd.read_csv(IRIS)
X = iris.iloc[:, [0, 1, 2]].values
Y = iris.iloc[:, 3].values
print(get_best_k(X, Y, q_num=100))
# (0.59, 0.037738709088905156)

# 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
model = RidgeCV(alphas=[0.1, 10, 10000])

# 线性回归建模
model.fit(X, Y)

print('系数:', model.coef_)
# 系数: [-0.20370879  0.21952122  0.52216614]

print(model.intercept_)
# -0.24377819461092076
