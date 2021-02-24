# Author: HaoLin You
# Created: 2021-01-07
# Last Modified: 2021-01-07
# Version: 1.0.0
# Description: 本节拟通过一个简单的例子说明用Python进行预测的主要步骤，旨在让各位读者了解用Python进行预测的基本过程
#   本节使用 wine_ind 数据集，它表示从1980年1月到1994年8月，葡萄酒生产商销售的容量不到1升的澳大利亚葡萄酒的总量

from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot as plt
from utils.data_path import WINE_IND
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np

# Mac 系统，使用如下设置可解决图表中文乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

"""
 基于 wine_ind 数据集，使用 statsmodels.graphics.tsaplots 
 模块下面的 plot_acf 函数查看 wine_ind 数据的自相关性
"""
wine_ind = pd.read_csv(WINE_IND)
wine_ind = pd.concat([wine_ind, wine_ind.月份.str.split('-', expand=True)], axis=1)
wine_ind = wine_ind.drop(columns=['月份']).rename(columns={0: "年份", 1: "月份"})
wine_ind['月份'] = [int(x) for x in wine_ind.月份]
plot_acf(wine_ind.销量, lags=100, title="wine_ind 自相关性").show()

"""
 我们可以将 1980年 到 1993年 每年按月的曲线画在一张图中
"""
plt.figure(figsize=(10, 5))

for __year__ in pd.unique(wine_ind.年份):
    df_row = wine_ind.loc[wine_ind.年份 == __year__, ]
    plt.plot(df_row.月份.values, df_row.销量.values, 'o--', label=__year__)

x = np.arange(1, 13)
plt.legend(ncol=7)
# 添加辅助线
plt.plot(x, 1500*x+17000, 'b--', linewidth=3)
plt.xlabel("月份", fontsize=12)
plt.ylabel("销量", fontsize=12)
plt.title("葡萄酒按月销量变化", fontsize=14)
plt.show()

"""
 数据转换
"""
sales_list = [None]*12 + wine_ind.销量.tolist()
for loc in [1, 4, 6, 8, 12]:
    wine_ind['近'+str(loc)+'月_销量'] = sales_list[(12-loc):][0:len(wine_ind)]

wine_ind = wine_ind.dropna()

# 画出散点矩阵图
sns.pairplot(wine_ind, diag_kind='kde')
plt.show()

"""
 建立 销量 ~ 近12月_销量 的线性模型，通过cooks标准来计算每行记录对模型的影响程度
"""

fig, ax = plt.subplots(figsize=(12, 8))
# 使用普通最小二乘法拟合一条线
lm = sm.OLS(wine_ind.销量, sm.add_constant(wine_ind.近12月_销量)).fit()
sm.graphics.influence_plot(lm, alpha=0.05, ax=ax, criterion="cooks")
plt.show()

"""
 从图中可知 91号 和 135号 样本存在明显的异常
 现将这两个点在 value~r12_value 对应的散点图中标记出来
"""
abnormal_points = wine_ind.loc[[91, 135], ]
plt.figure(figsize=(8, 5))
plt.plot(wine_ind.近12月_销量, wine_ind.销量, 'o', c='black')
plt.scatter(abnormal_points.近12月_销量, abnormal_points.销量, marker='o', c='white', edgecolors='k', s=200)
plt.xlabel("近12月_销量")
plt.ylabel("销量")
plt.show()

"""
 91号和135号的点正是我们通过散点矩阵图发现的杠杆点。现将这两个样本从 pdata 中去掉
"""

wine_ind = wine_ind.drop(index=[91, 135])

"""
 根据上一步得到的基础数据 wine_ind，提取其前150行数据作为训练集，余下的部分作为测试集，进行分区
"""
wine_ind = wine_ind.reset_index().drop(columns='index')
train_set = wine_ind.loc[0:149, ]
test_set = wine_ind.loc[149:, ]
X = np.column_stack((train_set.月份, train_set.近1月_销量, train_set.近4月_销量,
                     train_set.近6月_销量, train_set.近8月_销量, train_set.近12月_销量))
X = sm.add_constant(X)
model = sm.OLS(train_set.销量, X).fit()
print(model.summary())

"""
 重新构建模型，代码如下
"""

X = np.column_stack((train_set.月份, train_set.近4月_销量, train_set.近8月_销量, train_set.近12月_销量))
X = sm.add_constant(X)
model = sm.OLS(train_set.销量, X).fit()
print(model.summary())

"""
 尝试使用非线性的思路来进一步拟合模型，在模型中加入 x2(近4月_销量) 对应的二次项、三次项，重新建模
"""
X = np.column_stack((train_set.月份, train_set.近4月_销量,
                    train_set.近4月_销量**2,
                    train_set.近4月_销量**3,
                    train_set.近8月_销量,
                    train_set.近12月_销量))
X = sm.add_constant(X)
model = sm.OLS(train_set.销量, X).fit()
print(model.summary())

"""
 将Model作为预测模型，对预测数据集 test_set 进行预测
"""

X = np.column_stack((test_set.月份,
                    test_set.近4月_销量,
                    test_set.近4月_销量**2,
                    test_set.近4月_销量**3,
                    test_set.近8月_销量,
                    test_set.近12月_销量))
X = sm.add_constant(X)
y_pred = model.predict(X)
diff = np.abs(test_set.销量 - y_pred)/test_set.销量
print(diff)

# 统计预测结果
print(diff.describe())
