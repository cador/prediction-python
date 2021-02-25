from utils import *

iris = pd.read_csv(IRIS)
x1 = iris['Sepal.Length'].values
x2 = iris['Sepal.Width'].values
x3 = iris['Petal.Length'].values
y = iris['Petal.Width'].values

f_a0, f_b0, f_c0, f_d0 = 0, 0, 0, 0
min_mse = 1e10
k = 55
for a in range(k + 1):
    for b in range(k + 1):
        for c in range(k + 1):
            for d in range(k + 1):
                a0 = 2.0 * a / k - 1
                b0 = 2.0 * b / k - 1
                c0 = 2.0 * c / k - 1
                d0 = 2.0 * d / k - 1
                y0 = a0 + b0 * x1 + c0 * x2 + d0 * x3
                mse = np.mean((y - y0) ** 2)
                if mse < min_mse:
                    min_mse = mse
                    f_a0 = a0
                    f_b0 = b0
                    f_c0 = c0
                    f_d0 = d0

print(min_mse)
# 0.03607966942148759

print(f_a0, f_b0, f_c0, f_d0)
# (-0.34545454545454546, -0.19999999999999996, 0.23636363636363633, 0.5272727272727273)
