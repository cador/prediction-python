from utils import *

# 准备基础数据
iris = pd.read_csv(IRIS)
x, y = iris.drop(columns=['Species', 'Petal.Width']), iris['Petal.Width']

# 标准化处理
x = x.apply(lambda v: (v - np.mean(v)) / np.std(v))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
N = x_train.shape[0]
y_train = np.array([0] + list(y_train.values))

# 设置参数sigma
sigma = 10
omiga = np.zeros((N, N))
for i in range(N):
    xi = x_train.iloc[i, :]
    omiga[i, :] = np.exp(-np.sum((x_train - xi) ** 2, axis=1) / sigma ** 2)

# 设置平衡参数gama
gama = 10

# 构建矩阵A
A = (omiga + (1 / gama) * np.identity(N))
A = np.c_[[1] * N, A]
A = np.r_[np.array([[0] + [1] * N]), A]

# 求b和alpha参数
b_alpha = np.matmul(np.linalg.inv(A), y_train)
b = b_alpha[0]
alpha = b_alpha[1:]

# 基于 x_test 进行预测
y_pred = []
for i in range(x_test.shape[0]):
    xi = x_test.iloc[i, :]
    t0 = np.exp(-np.sum((x_train - xi) ** 2, axis=1) / sigma ** 2)
    y_pred.append(np.matmul(t0, alpha) + b)

# 误差平方和
print(np.sum((np.array(y_pred) - y_test) ** 2))
# 1.7705611499526264
