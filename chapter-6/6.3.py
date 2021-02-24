import pandas as pd
import numpy as np
from utils.data_path import IRIS

iris = pd.read_csv(IRIS)
x = iris.iloc[:, [0, 1, 2]]
y = iris.iloc[:, 3]
x = x.apply(lambda __x: (__x - np.mean(__x)) / np.sqrt(np.sum((__x - np.mean(__x)) ** 2))).values
y = (y - np.mean(y)).values

# 活动变量下标集合
m = x.shape[1]
active = []
max_steps = m + 1

# 初始化回归系数矩阵
beta = np.zeros((max_steps, m))
eps = 2.220446e-16

# 非活动变量与残差的相关系数
c = []
sign = []

# 非活动变量下标集合
im = range(m)
inactive = range(m)
k = 0

# 计算y与x的相关性
c_vec = np.matmul(y.T, x)

# 被忽略的变量下标集合
ignores = []

while k < max_steps and len(active) < m:
    c = c_vec[inactive]
    c_max = np.max(np.abs(c))
    if c_max < eps * 100:
        print("最大的相关系数为0，退出循环\n")
        break
    new = np.abs(c) >= c_max - eps
    c = c[np.logical_not(new)]
    new = np.array(inactive)[new]
    for i_new in new:
        if np.linalg.matrix_rank(x[:, active + [i_new]]) == len(active):
            ignores.append(i_new)
        else:
            active.append(i_new)
            sign.append(np.sign(c_vec[i_new]))

    active_len = len(active)
    exclude = active + ignores
    inactive = []
    t0 = [inactive.append(v) if i not in exclude else None for i, v in enumerate(im)]
    xa = x[:, active] * sign
    one_A = [1] * active_len
    A = np.matmul(np.matmul(one_A, np.linalg.inv(np.matmul(xa.T, xa))), one_A) ** (-0.5)
    w = np.matmul(A * np.linalg.inv(np.matmul(xa.T, xa)), one_A)
    if active_len >= m:
        gam_hat = c_max / A
    else:
        a = np.matmul(np.matmul(x[:, inactive].T, xa), w)
        gam = np.array([(c_max - c) / (A - a), (c_max + c) / (A + a)])
        gam_hat = np.min([np.min(gam[gam > eps]), c_max / A])

    b1 = beta[k, active]
    z1 = np.array(-b1 / (w * sign))
    z_min = np.min(z1[z1 > eps].tolist() + [gam_hat])
    gam_hat = z_min if z_min < gam_hat else gam_hat
    beta[k + 1, active] = beta[k, active] + gam_hat * w * sign
    c_vec = c_vec - gam_hat * np.matmul(np.matmul(xa, w).T, x)
    k = k + 1

print(beta)
# array([[ 0.        ,  0.        ,  0.        ],
#       [ 0.        ,  0.        ,  8.65652655],
#       [ 0.        ,  0.27627203,  8.93279858],
#       [-2.09501133,  1.18554279, 11.29305357]])
