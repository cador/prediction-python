from utils import *


def gbrt_build(x, y, con_same=5, max_iter=10000, shrinkage=0.0005):
    """
    建立函数构建 GBRT 模型
    :param x : 输入数据，解释变量
    :param y : 输出数据，响应变量
    :param con_same : 当连续 con_same 次得到的残差平方和相等时算法终止
    :param max_iter : 迭代次数的上限
    :param shrinkage : 缩放因子
    """
    # 使平方损失函数最小化的常数值为对应数据的平均值，即以均值初始化f0
    f0 = np.mean(y)
    # 初始化变量
    __rss = []
    __model_list = [f0]
    # 进入循环，当连续 con_same 次，得到的残差平方和相等或超过最大迭代次数时，终止算法
    for i in range(max_iter):
        # 计算负梯度，当损失函数为平方损失函数时，负梯度即为残差
        re_val = y - f0
        # 根据残差学习一棵回归树，设置分割点满足的最小样本量为 30
        __clf = tree.DecisionTreeRegressor(min_samples_leaf=30)
        __clf = __clf.fit(x, re_val)
        # 更新回归树，并生成估计结果
        __model_list.append(__clf)
        f0 = f0 + shrinkage * __clf.predict(x)
        # 统计残差平方和
        __rss.append(np.sum((f0 - y) ** 2))
        # 判断是否满足终止条件
        if len(__rss) >= con_same and np.std(__rss[(len(__rss) - con_same + 1):len(__rss)]) == 0:
            print("共迭代", i + 1, "次，满足终止条件退出迭代！")
            break
    return __rss, __model_list


iris = pd.read_csv(IRIS)
X, Y = iris.drop(columns=['Species', 'Petal.Width']), iris['Petal.Width']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)
rss, model_list = gbrt_build(x_train, y_train)

# 查看rss的统计信息
print(pd.Series(rss).describe())
# count    10000.000000
# mean         9.443136
# std         11.613604
# min          2.954312
# 25%          3.230344
# 50%          3.919855
# 75%          9.398138
# max         59.853140
# dtype: float64

# 根据 rss 绘制曲线，以直观观察残差平方和的变化趋势
plt.plot(range(10000), rss[0:10000], '-', c='black', linewidth=3)
plt.xlabel("迭代次数", fontsize=15)
plt.ylabel("RSS", fontsize=15)
plt.show()


def gbrt_predict(x, gml_list, shrinkage):
    """
    建立预测函数，对新数据进行预测
    :param x : 进行预测的新数据
    :param gml_list : 即 GBRT 的模型列表
    :param shrinkage : 训练模型时，指定的shrinkage参数
    """
    f0 = gml_list[0]
    for i in range(1, len(gml_list)):
        f0 = f0 + shrinkage * gml_list[i].predict(x)
    return f0


print(np.sum((y_test - gbrt_predict(x_test, model_list, 0.0005)) ** 2))
# 1.3384897597030645

clf = tree.DecisionTreeRegressor(min_samples_leaf=30)
clf = clf.fit(x_train, y_train)
print(np.sum((y_test - clf.predict(x_test)) ** 2))
# 1.5676935145052517
