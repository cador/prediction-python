import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_path import CANADA
import statsmodels.tsa.stattools as stat

src_canada = pd.read_csv(CANADA)
val_columns = ['e', 'prod', 'rw', 'U']
v_std = src_canada[val_columns].apply(lambda x: np.std(x)).values
v_mean = src_canada[val_columns].apply(lambda x: np.mean(x)).values
canada = src_canada[val_columns].apply(lambda x: (x - np.mean(x)) / np.std(x))
train = canada.iloc[0:-8]

for col in val_columns:
    _, p_value, _, _, _, _ = stat.adfuller(train[col], 1)
    print("指标", col, "单位根检验的p值为： ", p_value)

# 指标 e 单位根检验的p值为：  0.9255470701604621
# 指标 prod 单位根检验的p值为：  0.9479865217623266
# 指标 rw 单位根检验的p值为：  0.0003397509672252013
# 指标 U 单位根检验的p值为：  0.19902577436726288

# 由于这4个指标都不平稳，因此需要进行合适的差分运算
train_diff = train.apply(lambda x: np.diff(x), axis=0)

for col in val_columns:
    _, p_value, _, _, _, _ = stat.adfuller(train_diff[col], 1)
    print("指标", col, "单位根检验的p值为： ", p_value)

# 指标 e 单位根检验的p值为：  0.00018806258268032046
# 指标 prod 单位根检验的p值为：  7.3891405425103595e-09
# 指标 rw 单位根检验的p值为：  1.254497644415662e-06
# 指标 U 单位根检验的p值为：  7.652834648091671e-05

# 模型阶数从1开始逐一增加
rows, cols = train_diff.shape
aic_list = []
lm_list = []

for p in range(1, 11):
    base_data = None
    for i in range(p, rows):
        tmp_list = list(train_diff.iloc[i]) + list(train_diff.iloc[i - p:i].values.flatten())
        if base_data is None:
            base_data = [tmp_list]
        else:
            base_data = np.r_[base_data, [tmp_list]]
    X = np.c_[[1] * base_data.shape[0], base_data[:, cols:]]
    Y = base_data[:, 0:cols]
    coef_matrix = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)
    aic = np.log(np.linalg.det(np.cov(Y - np.matmul(X, coef_matrix), rowvar=False))) + 2 * (coef_matrix.shape[0] - 1) ** 2 * p / base_data.shape[0]
    aic_list.append(aic)
    lm_list.append(coef_matrix)

# 对比查看阶数和AIC值
pd.DataFrame({"P": range(1, 11), "AIC": aic_list})
#   P      AIC
# 0  1   -19.996796
# 1  2   -17.615455
# 2  3   -9.407306
# 3  4   6.907540
# 4  5   34.852248
# 5  6   77.620404
# 6  7   138.382810
# 7  8   220.671801
# 8  9   328.834718
# 9  10  466.815468

p = np.argmin(aic_list) + 1
n = rows
pred_df = None
for i in range(8):
    pred_data = list(train_diff.iloc[n + i - p:n + i].values.flatten())
    pred_val = np.matmul([1] + pred_data, lm_list[p - 1])
    # 使用逆差分运算，还原预测值
    pred_val = train.iloc[n + i, :] + pred_val
    if pred_df is None:
        pred_df = [pred_val]
    else:
        pred_df = np.r_[pred_df, [pred_val]]

    # 为train增加一条新记录
    train = train.append(canada[n + i + 1:n + i + 2], ignore_index=True)
    # 为train_diff增加一条新记录
    df = pd.DataFrame(list(canada[n + i + 1:n + i + 2].values - canada[n + i:n + i + 1].values), columns=canada.columns)
    train_diff = train_diff.append(df, ignore_index=True)

pred_df = pred_df * v_std + v_mean

# 分析预测残差情况
pred_df - src_canada[canada.columns].iloc[-8:].values
# array([[ 0.20065717, -0.7208273 ,  0.08095578, -0.18725653],
#        [ 0.03650856, -0.08061888,  0.05900709, -0.22667618],
#        [ 0.03751544, -0.87174186,  0.17291551,  0.10381011],
#        [-0.04826459, -0.06498827,  0.45879439,  0.34885492],
#        [-0.15647981, -0.6096229 , -1.1219943 , -0.12520269],
#        [ 0.51480518, -0.51864268,  0.7123945 , -0.2760806 ],
#        [ 0.32312138, -0.06077591, -0.14816924, -0.39923473],
#        [-0.34031027,  0.78080541,  1.31294708,  0.01779691]])

# 统计预测百分误差率分布
pd.Series((np.abs(pred_df - src_canada[canada.columns].iloc[-8:].values) * 100 / src_canada[canada.columns].iloc[-8:].values).flatten()).describe()
# count    32.000000
# mean      0.799252
# std       1.551933
# min       0.003811
# 25%       0.018936
# 50%       0.111144
# 75%       0.264179
# max       5.760963
# dtype: float64

m = 16
xts = src_canada[['year', 'season']].iloc[-m:].apply(lambda x: str(x[0]) + '-' + x[1], axis=1).values
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
index = 0
for ax in axes.flatten():
    ax.plot(range(m), src_canada[canada.columns].iloc[-m:, index], '-', c='lightgray', linewidth=2, label="real")
    ax.plot(range(m - 8, m), pred_df[:, index], 'o--', c='black', linewidth=2, label="predict")
    ax.set_xticks(range(m))
    ax.set_xticklabels(xts, rotation=50)
    ax.set_ylabel("$" + canada.columns[index] + "$", fontsize=14)
    ax.legend()
    index = index + 1
plt.tight_layout()
plt.show()
