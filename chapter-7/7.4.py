from utils import *

# 准备基础数据
iris = pd.read_csv(IRIS)
x, y = iris.drop(columns=['Species', 'Petal.Width']), iris['Petal.Width']

# 标准化处理
x = x.apply(lambda v: (v - np.mean(v)) / np.std(v))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

# 初始化参数
n = x_train.shape[0]
epsilon = 1e-3
theta1 = 1
theta2 = 1
theta3 = 1
learn_rate = 0.005


def delta(bgc, __delta, __y):
    bgc_inv = np.linalg.inv(bgc)
    a = np.sum(np.diag(np.matmul(bgc_inv, __delta)))
    b = np.matmul(np.matmul(__y, np.matmul(np.matmul(bgc_inv, __delta), bgc_inv)), __y)
    return 0.5 * (a - b)


def big_c(data, t1, t2, t3):
    rows = data.shape[0]
    tmp = np.zeros((rows, rows))
    for e in range(rows):
        x_tmp = data.iloc[e, :]
        tmp[e, :] = np.exp(2 * t2) * np.exp(-np.sum((data - x_tmp) ** 2, axis=1) / (2 * np.exp(2 * t1)))
    return tmp + np.identity(rows) * np.exp(2 * t3)


for i in range(1000):
    big_c_out = big_c(x_train, theta1, theta2, theta3)
    # 更新theta1
    delta1 = np.zeros((n, n))
    for j in range(n):
        xi = x_train.iloc[j, :]
        deltaX = (x_train - xi) ** 2
        rs_obj = np.sum(deltaX, axis=1)
        delta1[j, :] = (np.exp(2 * theta2) ** 2) * np.exp(-rs_obj / (2 * np.exp(2 * theta1))) * rs_obj / (2 * np.exp(2 * theta1))

    delta1 = delta(big_c_out, delta1, y_train)
    theta1 = theta1 - learn_rate * delta1

    # 更新theta2
    delta2 = np.zeros((n, n))
    for j in range(n):
        xi = x_train.iloc[j, :]
        deltaX = (x_train - xi) ** 2
        delta2[j, :] = 2 * (np.exp(2 * theta2) ** 2) * np.exp(-np.sum(deltaX, axis=1) / (2 * np.exp(2 * theta1)))

    delta2 = delta(big_c_out, delta2, y_train)
    theta2 = theta2 - learn_rate * delta2

    # 更新theta3
    delta3 = np.identity(n) * np.exp(2 * theta3)
    delta3 = delta(big_c_out, delta3, y_train)
    theta3 = theta3 - learn_rate * delta3
    print(i, "---delta1:", delta1, "delta2:", delta2, "delta3:", delta3)

    # 当超参数的变化量绝对值的最大值小于给定精度时，退出循环
    if np.max(np.abs([delta1, delta2, delta3])) < epsilon:
        break

# 0 ---delta1: -15.435977359055135 delta2: 28.942308902124964 delta3: 47.0407871507001
# 1 ---delta1: -11.20212191847591 delta2: 20.269730245089818 delta3: 46.90575619288189
# 2 ---delta1: -9.326096699821793 delta2: 15.67459048871474 delta3: 46.59119394310448
# 3 ---delta1: -8.217168550831616 delta2: 12.588401264999455 delta3: 46.12465918227182
# ......
# 138 ---delta1: -0.0006823077050732707 delta2: -0.0009742826717165087 delta3: -2.6515727903131392e-05

# 求得的3个超参数分别为
print(theta1, theta2, theta3)
# (1.4767280756916963, 0.5247171125067923, -1.7670980634788505)

# 进行预测并计算残差平方和
bigC = big_c(x_train, theta1, theta2, theta3)
alpha = np.matmul(np.linalg.inv(bigC), y_train)
y_pred = []
y_sigma = []
tn = x_test.shape[0]
for j in range(tn):
    xi = x_test.iloc[j, :]
    deltaX = (x_train - xi) ** 2
    t0 = np.exp(2 * theta2) * np.exp(-np.sum(deltaX, axis=1) / (2 * np.exp(2 * theta1)))
    y_pred.append(np.matmul(t0, alpha))
    y_sigma.append(np.sqrt(np.exp(2 * theta2) - np.matmul(np.matmul(t0, np.linalg.inv(bigC)), t0)))

# 最终得到的残差平方和为
print(np.sum((y_test.values - y_pred) ** 2))
# 2.081954371791342

print(pd.DataFrame({'y_test': y_test, 'y_pred': y_pred, 'sigma': y_sigma}).head())
#     y_test    y_pred      sigma
# 14    0.2  0.170740   0.114043
# 98    1.1  0.820464   0.048525
# 75    1.4  1.410814   0.047854
# 16    0.4  0.201179   0.067488
# 131   2.0  2.145182   0.151244
