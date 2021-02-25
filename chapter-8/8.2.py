from utils import *

ss_data = pd.read_csv(SUN_SPOT)
rows = len(ss_data)

# 使用后50年的数据进行验证，以前的数据用于建模
train_data = ss_data[0:(rows - 50)]
n = train_data.shape[0]

# 设置候选门限值
thresholdV = train_data.sunspot.sort_values().values[np.int32(np.arange(30, 70, 3) / 100 * n)]

# 设置最大门限延迟量d_max、自回归最大阶数、默认最小AIC值
d_max = 5
p_max = 5
min_aic = 1e+10


def get_model_info(ts_obj, d, p, r, is_up=True):
    """在指定门限延迟量、阶数及门限值的前提下，返回对应自回归模型AIC值和自回归系数"""
    if is_up:
        dst_set = np.where(ts_obj > r)[0] + d
    else:
        dst_set = np.where(ts_obj <= r)[0] + d

    tmp_data = None

    # 重建基础数据集
    # xt=a0+a1*x(t-1)+...+ap*x(t-p)
    for i in dst_set:
        if p < i < len(ts_obj):
            if tmp_data is None:
                tmp_data = [ts_obj[(i - p):(i + 1)]]
            else:
                tmp_data = np.r_[tmp_data, [ts_obj[(i - p):(i + 1)]]]
    x = np.c_[[1] * tmp_data.shape[0], tmp_data[:, 0:p]]
    coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), tmp_data[:, p])
    epsilon = tmp_data[:, p] - np.matmul(x, coef)
    aic = tmp_data.shape[0] * np.log(np.var(epsilon)) + 2 * (p + 2)
    return {"aic": aic, "coef": coef}


# 选择最优参数
a_tsv, a_d, a_p1, a_p2, coef1, coef2 = 0, 0, 0, 0, 0, 0
for tsv in thresholdV:
    for d in range(1, d_max + 1):
        for p1 in range(1, p_max + 1):  # <= r
            model1 = get_model_info(train_data.sunspot.values, d, p=p1, r=tsv, is_up=False)
            for p2 in range(1, p_max + 1):  # > r
                model2 = get_model_info(train_data.sunspot.values, d, p=p2, r=tsv, is_up=True)
                if model1['aic'] + model2['aic'] < min_aic:
                    min_aic = model1['aic'] + model2['aic']
                    a_tsv = tsv
                    a_d = d
                    a_p1 = p1
                    a_p2 = p2
                    coef1 = model1['coef']
                    coef2 = model2['coef']
                    print(min_aic)

# 1891.4713402264924
# 1755.538487229318
# ......
# 1613.7875399449235
# 1612.4584851226264


pred_data = []

for k in range(rows - 50, rows):
    t0 = ss_data.sunspot.values[k - a_d]
    if t0 <= a_tsv:
        pred_data.append(np.sum(np.r_[1, ss_data.sunspot.values[(k - a_p1):k]] * coef1))
    else:
        pred_data.append(np.sum(np.r_[1, ss_data.sunspot.values[(k - a_p2):k]] * coef2))

plt.figure(figsize=(10, 6))
plt.plot(range(rows)[-100:rows], ss_data.sunspot[-100:rows], '-', c='black', linewidth=2, label="真实值")
plt.plot(range(rows)[-50:rows], pred_data, 'b--', label="预测值")
plt.xticks(range(rows)[-100:rows][::15], ss_data.year[-100:rows][::15], rotation=50)
plt.xlabel("$year$", fontsize=15)
plt.ylabel("$sunspot$", fontsize=15)
plt.legend()
plt.show()
