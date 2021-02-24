from scipy import stats
from scipy.spatial.distance import correlation as d_cor
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from utils.data_path import IRIS, WINE
from utils.udf import gains, disc, gains_ratio, eval_func

# Mac 系统，使用如下设置可解决图表中文乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

iris = pd.read_csv(IRIS)
print(stats.pearsonr(iris['Sepal.Length'], iris['Petal.Length']))
# (0.8717537758865832, 1.0386674194497583e-47)

x = np.linspace(-1, 1, 50)
y = x ** 2
plt.plot(x, y, 'o')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

print(stats.pearsonr(x, y))
# (-1.9448978228488063e-16, 1.0)

print(d_cor(x, y))
# 1.0

iris.columns = ['_'.join(x.split('.')) for x in iris.columns]
anova_lm(ols('Sepal_Width~C(Species)', iris).fit())
print(iris.head())

for col in iris.columns[0:-1]:
    iris[col] = disc(iris.Species, iris[col]).astype("int")
print(iris.head())

iris.columns = ["S1", "S2", "P1", "P2", "Species"]
print(gains(iris['Species'], iris['S1']))
# 0.5572326878069265
print(gains(iris['Species'], iris['S2']))
# 0.28312598916883114
print(gains(iris['Species'], iris['P1']))
# 0.9182958340544892
print(gains(iris['Species'], iris['P2']))
# 0.9182958340544892

iris['P2'] = range(iris.shape[0])
# 计算信息增益，并排序
print(gains(iris['Species'], iris['S1']))
# 0.5572326878069265
print(gains(iris['Species'], iris['S2']))
# 0.28312598916883114
print(gains(iris['Species'], iris['P1']))
# 0.9182958340544892
print(gains(iris['Species'], iris['P2']))
# 1.5849625007211559
# 重要性次序为：P2 > P1 > S1 > S2

# 计算信息增益率，并排序
print(gains_ratio(iris['Species'], iris['S1']))
# 0.5762983610929974
print(gains_ratio(iris['Species'], iris['S2']))
# 0.35129384185463564
print(gains_ratio(iris['Species'], iris['P1']))
# 0.9999999999999999
print(gains_ratio(iris['Species'], iris['P2']))
# 0.21925608713979675
# 重要性次序为：P1 > S1 > S2 > P2

iris = pd.read_csv(IRIS)

for col in iris.columns[:-1]:
    iris[col] = disc(iris.Species, iris[col]).astype("int")

iris['D'] = 1
chi_data = np.array(iris.pivot_table(values='D', index='Sepal.Width', columns='Species', aggfunc='sum'))
chi = chi2_contingency(chi_data)
print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s' % chi)
# chisq-statistic=57.1155, p-value=0.0000, df=2 expected_frep=[[37.66 37.66 37.66]
# [12.33 12.33 12.33]]

df = pd.read_csv(WINE, header=None)
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
X, Y = df.drop(columns='Class label'), df['Class label']
forest = RandomForestClassifier(n_estimators=10000, random_state=0)
forest.fit(X, Y)
importance = forest.feature_importances_
indices = np.argsort(importance)[::-1]
for e in range(X.shape[1]):
    print("%2d) %-*s %f" % (e + 1, 30, X.columns[indices[e]], importance[indices[e]]))

# 1) Proline                        0.172933
# 2) Color intensity                0.159572
# 3) Flavanoids                     0.158639
# 4) Alcohol                        0.122553
# 5) OD280/OD315 of diluted wines   0.117285
# 6) Hue                            0.082196
# 7) Total phenols                   0.052964
# 8) Magnesium                      0.030679
# 9) Malic acid                     0.030567
# 10) Alcalinity of ash              0.026736
# 11) Proanthocyanins                0.021301
# 12) Ash                            0.013659
# 13) Nonflavanoid phenols           0.010917

plt.title('特征重要性')
plt.bar(range(X.shape[1]), importance[indices], color='black', align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()

out = []
for i in range(10000):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=i)
    org_ratio = eval_func(X_train, Y_train, X_test, Y_test)
    eval_list = []
    for col in X_train.columns:
        new_train = X_train.copy()
        new_train[col] = random.choice(range(new_train.shape[0]))
        decrease = org_ratio - eval_func(new_train, Y_train, X_test, Y_test)
        eval_list.append(decrease if decrease > 0 else 0)
    out.append(eval_list)

importance = pd.DataFrame(np.array(out)).apply(lambda __x: np.mean(__x), axis=0).values
indices = np.argsort(importance)[::-1]

for e in range(X.shape[1]):
    print("%2d) %-*s %f" % (e + 1, 30, X.columns[indices[e]], importance[indices[e]]))

# 1) Color intensity                0.024522
# 2) Flavanoids                     0.021774
# 3) Proline                        0.014481
# 4) Alcohol                        0.009896
# 5) OD280/OD315 of diluted wines   0.009563
# 6) Ash                            0.009431
# 7) Hue                            0.009028
# 8) Total phenols                  0.008741
# 9) Alcalinity of ash              0.008669
# 10) Nonflavanoid phenols           0.008628
# 11) Malic acid                     0.008496
# 12) Magnesium                      0.008493
# 13) Proanthocyanins                0.008333

plt.title('特征重要性')
plt.bar(range(X.shape[1]), importance[indices], color='black', align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()
