import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.data_path import AGR_INDEX
import statsmodels.tsa.stattools as stat
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Mac 系统，使用如下设置可解决图表中文乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载基础数据
ts_data = pd.read_csv(AGR_INDEX)
rows = ts_data.shape[0]
plt.figure(figsize=(10, 6))
plt.plot(range(rows), ts_data.agr_index, '-', c='black', linewidth=3)
plt.xticks(range(rows)[::3], ts_data.year[::3], rotation=50)
plt.xlabel("$year$", fontsize=15)
plt.ylabel("$agr\\_index$", fontsize=15)
plt.show()

# 此处预留10年的数据进行验证
test_data = ts_data[(rows - 10):rows]
train_data = ts_data[0:(rows - 10)]

# 进行d阶差分运算
d = 1
z = []
for t in range(d, train_data.shape[0]):
    tmp = 0
    for i in range(0, d + 1):
        tmp = tmp + (-1) ** i * (np.math.factorial(d) / (np.math.factorial(i) * np.math.factorial(d - i))) * train_data.iloc[t - i, 1]
    z.append(tmp)

# 使用单位根检验差分后序列的平稳性
print(stat.adfuller(z, 1))
# (-3.315665263756724,
# 0.014192594291845282,
# 1,
# 24,
# {'1%': -3.7377092158564813,
#  '5%': -2.9922162731481485,
#  '10%': -2.635746736111111},
# 161.9148847383757)
lb_rs = pd.DataFrame(lb_test(z, boxpierce=True, return_df=True, lags=min(10, len(z)//5)))

plt.plot(lb_rs.lb_pvalue, 'o-', c='black', label="LB-p值")
plt.plot(lb_rs.bp_pvalue, 'o--', c='black', label="BP-p值")
plt.legend()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
ax0, ax1 = axes.flatten()
plot_acf(z, ax=ax0, lags=5, alpha=0.05)
plot_pacf(z, ax=ax1, lags=5, alpha=0.05)
plt.show()

# 基于最小化残差平方和的假设，使用梯度下降法拟合未知参数
miu = np.mean(z)
print(miu)
# 2.3538461538461535

theta1 = 0.5
alpha = 0.0001
epsilon_theta1 = 0
errorList = []
for k in range(60):
    epsilon = 0
    error = 0
    for i in range(len(z)):
        epsilon_theta1 = epsilon + theta1 * epsilon_theta1
        theta1 = theta1 - alpha * 2 * (z[i] - miu + theta1 * epsilon) * epsilon_theta1
        epsilon = z[i] - miu + theta1 * epsilon
        error = error + epsilon ** 2

    errorList.append(error)
    print("iter:", k, " error:", error)
    # 当连续两次残差平方和的差小于1e-5时，退出循环
    if len(errorList) > 2 and np.abs(errorList[-2] - errorList[-1]) < 1e-5:
        break

# iter: 0  error: 2158.5528412123554
# iter: 1  error: 1629.117571439535
# iter: 2  error: 1378.697932325996
# ......
# iter: 14  error: 873.0924002584927
# ......
# iter: 37  error: 863.6232496284028
# iter: 38  error: 863.6232390131801
# iter: 39  error: 863.623233001729

print(theta1)
# -0.7940837640329033

error = []
epsilon = 0
for i in range(len(z)):
    epsilon = z[i] - miu + theta1 * epsilon
    error.append(epsilon)

# 使用 Ljung-Box 检验 error 序列是否为白噪声
lb_rs = pd.DataFrame(lb_test(error, boxpierce=True, return_df=True, lags=min(10, len(error)//5)))
plt.plot(lb_rs.lb_pvalue, 'o-', c='black', label="LB-p值")
plt.plot(lb_rs.bp_pvalue, 'o--', c='black', label="BP-p值")
plt.legend()
plt.show()

# 基于该模型对差分后的序列进行预测
predX = miu + np.mean(error) - theta1 * epsilon
print(predX)
# 4.745789901194965

# 由于经过1阶差分的运算，所以此处需要进行差分的逆运算，以计算原始序列对应的预测值
org_predX = train_data.iloc[-1, 1] + predX
print(org_predX)
# 165.94578990119496

# 对超过1期的预测值，统一为 predXt
predXt = org_predX + 2.353846 + 1.7940838 * np.mean(error)
print(predXt)
# 168.3897028849476

# 绘制出原始值和预测值
plt.figure(figsize=(10, 6))
plt.plot(range(rows), ts_data.agr_index, '-', c='black', linewidth=3)
plt.plot(range(train_data.shape[0], ts_data.shape[0]), [org_predX] + [predXt] * 9, 'o', c='gray')
plt.xticks(range(rows)[::3], ts_data.year[::3], rotation=50)
plt.xlabel("$year$", fontsize=15)
plt.ylabel("$agr\\_index$", fontsize=15)
plt.show()

pred_x = []
for i in range(10):
    pred_val = miu + np.mean(error) - theta1 * epsilon
    if i == 0:
        org_pred_x = train_data.iloc[-1, 1] + pred_val
    else:
        org_pred_x = test_data.iloc[i - 1, 1] + pred_val

    pred_x.append(org_pred_x)
    epsilon = test_data.iloc[i, 1] - org_pred_x

plt.figure(figsize=(10, 6))
plt.plot(range(rows), ts_data.agr_index, '-', c='black', linewidth=3)
plt.plot(range(train_data.shape[0], ts_data.shape[0]), pred_x, 'o--', c='red')
plt.xticks(range(rows)[::3], ts_data.year[::3], rotation=50)
plt.xlabel("$year$", fontsize=15)
plt.ylabel("$agr\\_index$", fontsize=15)
plt.show()
