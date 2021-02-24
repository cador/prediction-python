import pandas as pd
import numpy as np
from scipy.optimize import linprog
from utils.data_path import IRIS

iris = pd.read_csv(IRIS)
x = iris.iloc[:, [0, 1, 2]]
y = iris.iloc[:, 3]
tau = 0.35
n, m = x.shape
c = [0] * 2 * m + [tau] * n + [1 - tau] * n
A_eq = np.c_[x, -x, np.identity(n), -np.identity(n)]
b_eq = y
r = linprog(c, A_eq=A_eq, b_eq=b_eq, method='simplex')
# 求解的回归系数
print(r.x[0:3] - r.x[3:6])
# array([-0.18543956,  0.125,  0.48076923])
