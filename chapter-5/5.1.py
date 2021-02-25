from utils import *


def sample_split(df, __k):
    """
    本函数实现对样本的分割
    :param df:数据框对象
    :param __k:分割数量
    :return :返回类别数组
    """
    t0 = np.array(range(len(df))) % __k
    random.shuffle(t0)
    return t0


iris = pd.read_csv(IRIS)
X = iris[['Sepal.Length', 'Sepal.Width', 'Petal.Length']]
Y = iris['Petal.Width']

# 设置为10折交叉验证
k = 10
parts = sample_split(iris, k)

# 初始化最小均方误差 min_error
min_error = 1000

# 初始化最佳拟合结果 final_fit
final_fit = None

for i in range(k):
    reg = linear_model.LinearRegression()
    X_train = X.iloc[parts != i, ]
    Y_train = Y.loc[parts != i, ]
    X_test = X.iloc[parts == i, ]
    Y_test = Y.loc[parts == i, ]

    # 拟合线性回归模型
    reg.fit(X_train, Y_train)

    # 计算均方误差
    error = np.mean((Y_test.values - reg.predict(X_test)) ** 2)

    if error < min_error:
        min_error = error
        final_fit = reg

print(min_error)
# 0.014153864387035597

print(final_fit.coef_)
# array([[-0.20013407,  0.21165518,  0.51669477]])

# 2、使用一般方法得到的参数
reg = linear_model.LinearRegression()
reg.fit(X, Y)
print(reg.coef_)
# array([[-0.20726607,  0.22282854,  0.52408311]])
