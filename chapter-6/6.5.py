import pandas as pd
import numpy as np
from sklearn import linear_model
import random
from utils.data_path import IRIS

iris = pd.read_csv(IRIS)
x = iris.iloc[:, [0, 1, 2]].values
y = iris.iloc[:, 3].values

# 对x中的每个变量，随机替换5个较大值
row, col = x.shape
for k in range(col):
    x[random.sample(range(row), 5), k] = np.random.uniform(10, 30, 5)

clf = linear_model.LinearRegression()
clf.fit(x, y)
x_input = np.c_[x, [1] * row]
col = col + 1
beta = list(clf.coef_) + [clf.intercept_]
c = 0.8
k = 0

# 迭代法求解
while k < 100:
    y0 = np.matmul(x_input, beta)
    epsilon = y - y0
    delta = np.std(epsilon)
    epsilon = epsilon / delta
    fai_k = [0] * col
    fai_kD = np.zeros((col, col))
    for i in range(row):
        if np.abs(epsilon[i]) <= c:
            xi = x_input[i, :]
            fai_k = fai_k + np.sin(epsilon[i] / c) * xi
            fai_kD = fai_kD - np.cos(epsilon[i] / c) * np.array([h * xi for h in xi]) / (delta * c)

    b = np.matmul(np.linalg.inv(fai_kD), fai_k)
    beta = beta - b
    print(np.max(np.abs(b)))
    if np.max(np.abs(b)) < 1e-15:
        print("算法收敛，退出循环！")
        break

    k = k + 1

# 0.32420863302317215
# 0.10076592577534327
# 0.34463605932944774
# 0.002844992435885897
# 3.149072360695391e-06
# 3.9248910208582647e-10
# 2.5256493241063437e-13
# 1.720771316794861e-16
# 算法收敛，退出循环！

print(np.mean(np.abs(y - np.matmul(x_input, beta))))
# 0.34227957292615147

print(beta)
# array([-0.00282362, -0.00279034,  0.41039862, -0.32295107])

clf.fit(x, y)
print(list(clf.coef_) + [clf.intercept_])
# [0.0194587459398201, -0.021165276875958147, 0.1151460137114351, 0.6648607142724917]

print(np.mean(np.abs(clf.predict(x) - y)))
# 0.5064579193197599
