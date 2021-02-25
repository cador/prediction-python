from utils import *

data = web.DataReader('600519.ss', 'yahoo', dt.datetime(2014, 1, 1), dt.datetime(2019, 9, 30))
sub_data = data.iloc[:-30, :4]
for i in range(4):
    p_value = stat.adfuller(sub_data.values[:, i], 1)[1]
    print("指标 ", data.columns[i], " 单位根检验的p值为：", p_value)
# 指标  High  单位根检验的p值为： 0.9955202280850401
# 指标  Low  单位根检验的p值为： 0.9942509439755689
# 指标  Open  单位根检验的p值为： 0.9938548193990323
# 指标  Close  单位根检验的p值为： 0.9950049124079876

sub_data_diff1 = sub_data.iloc[1:, :].values - sub_data.iloc[:-1, :].values
for i in range(4):
    p_value = stat.adfuller(sub_data_diff1[:, i], 1)[1]
    print("指标 ", data.columns[i], " 单位根检验的p值为：", p_value)
# 指标  High  单位根检验的p值为： 0.0
# 指标  Low  单位根检验的p值为： 0.0
# 指标  Open  单位根检验的p值为： 0.0
# 指标  Close  单位根检验的p值为： 0.0

# 模型阶数从1开始逐一增加
rows, cols = sub_data_diff1.shape
aicList = []
lmList = []

for p in range(1, 11):
    baseData = None
    for i in range(p, rows):
        tmp_list = list(sub_data_diff1[i, :]) + list(sub_data_diff1[i - p:i].flatten())
        if baseData is None:
            baseData = [tmp_list]
        else:
            baseData = np.r_[baseData, [tmp_list]]
    X = np.c_[[1] * baseData.shape[0], baseData[:, cols:]]
    Y = baseData[:, 0:cols]
    coef_matrix = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)
    aic = np.log(np.linalg.det(np.cov(Y - np.matmul(X, coef_matrix), rowvar=False))) + 2 * (
                coef_matrix.shape[0] - 1) ** 2 * p / baseData.shape[0]
    aicList.append(aic)
    lmList.append(coef_matrix)

# 对比查看阶数和AIC
pd.DataFrame({"P": range(1, 11), "AIC": aicList})
#   P   AIC
# 0 1   13.580156
# 1 2   13.312225
# 2 3   13.543633
# 3 4   14.266087
# 4 5   15.512437
# 5 6   17.539047
# 6 7   20.457337
# 7 8   24.385459
# 8 9   29.438091
# 9 10  35.785909

p = np.argmin(aicList) + 1
n = rows
pred_df = None
for i in range(30):
    pred_data = list(sub_data_diff1[n + i - p:n + i].flatten())
    pred_val = np.matmul([1] + pred_data, lmList[p - 1])
    # 使用逆差分运算，还原预测值
    pred_val = data.iloc[n + i, :].values[:4] + pred_val
    if pred_df is None:
        pred_df = [pred_val]
    else:
        pred_df = np.r_[pred_df, [pred_val]]
    # 为sub_data_diff1增加一条新记录
    sub_data_diff1 = np.r_[sub_data_diff1, [data.iloc[n + i + 1, :].values[:4] - data.iloc[n + i, :].values[:4]]]

# 分析预测残差情况
(np.abs(pred_df - data.iloc[-30:data.shape[0], :4]) / data.iloc[-30:data.shape[0], :4]).describe()
#       High         Low         Open       Close
# count 30.000000   30.000000   30.000000   30.000000
# mean  0.010060    0.009380    0.005661    0.013739
# std   0.008562    0.009968    0.006515    0.013674
# min   0.001458    0.000115    0.000114    0.000130
# 25%   0.004146    0.001950    0.001653    0.002785
# 50%   0.007166    0.007118    0.002913    0.010414
# 75%   0.014652    0.012999    0.006933    0.022305
# max   0.039191    0.045802    0.024576    0.052800


plt.figure(figsize=(10, 7))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(range(30), data.iloc[-30:data.shape[0], i].values, 'o-', c='black')
    plt.plot(range(30), pred_df[:, i], 'o--', c='gray')
    plt.ylim(1000, 1200)
    plt.ylabel("$" + data.columns[i] + "$")
plt.show()
v = 100 * (1 - np.sum(np.abs(pred_df - data.iloc[-30:data.shape[0], :4]).values) / np.sum(data.iloc[-30:data.shape[0], :4].values))
print("Evaluation on test data: accuracy = %0.2f%% \n" % v)
# Evaluation on test data: accuracy = 99.03%
