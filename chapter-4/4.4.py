import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import copy
from utils.data_path import IRIS, CEMHT
from utils.udf import g, times, add, log, gen_full_tree_exp, gen_side_tree_exp, \
    random_get_tree, transform, plot_tree, gen_individuals, inter_cross, mutate, evaluation_regression, get_adjust

# Mac 系统，使用如下设置可解决图表中文乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

print(g(times, g(add, g(np.sin, 5), 10), g(log, 46)))
# 50.219458431129446

# 随机抽取 N 个特征下标
nMax = 10
N = random.choice(range(2, nMax + 1))

# 定义原始数据集中属性的下标1~8
feature_idx = range(1, 9)
select_feature_index = [random.choice(feature_idx) for i in range(N)] + [0] * int(2 ** np.ceil(np.log2(N)) - N)
random.shuffle(select_feature_index)
select_feature_index = ['X' + str(e) if e > 0 else '0' for e in select_feature_index]
print(select_feature_index)

tree_exp = gen_full_tree_exp(select_feature_index)
print(tree_exp)

N = random.choice(range(2, nMax + 1))
select_feature_index = ['X' + str(e) for e in [random.choice(feature_idx) for i in range(N)]]
print(select_feature_index)

tree_exp = gen_side_tree_exp(select_feature_index)
print(tree_exp)

iris = pd.read_csv(IRIS)
out = random_get_tree(iris, [1, 2, 3, 4])
print(out['tree_exp'])
print(out['f_value'])

exp_tmp = 'g(add,g(div,g(sub,g(sqrt,X1),X1),g(times,g(binary,X2),X3)),g(div,g(sigmoid,X3),g(add,X4,X2)))'
print(transform(exp_tmp))

plt.figure(figsize=(20, 11))
plot_tree(exp_tmp)
plt.show()

gen_out = gen_individuals(5, 4, iris, [1, 2, 3, 4])
for x in gen_out['df']:
    print("____________________________________________")
    print(x.head(2))


A = ['g(add,X1,X2)', 'g(log,X1)', 'g(add,g(log,X2),X3)']
B = ['g(pow2,X3)', 'g(add,g(inv,X1),g(log,X2))', 'g(log,g(tanh,X4))']
counter = 1
titles = ['个体A基因1', '个体A基因2', '个体A基因3', '个体B基因1', '个体B基因2', '个体B基因3']
plt.figure(figsize=(15, 8))
for e in A + B:
    plt.subplot(2, 3, counter)
    plot_tree(e, title=titles[counter - 1], node_size=500, font_size=10)
    counter += 1
plt.show()

inter_cross(None, [A, B], 1)
counter = 1
titles = ['个体A基因1', '个体A基因2', '个体A基因3', '个体B基因1', '个体B基因2', '个体B基因3']
plt.figure(figsize=(15, 8))
for e in A + B:
    plt.subplot(2, 3, counter)
    plot_tree(e, title=titles[counter - 1], node_size=500, font_size=10)
    counter += 1
plt.show()

# 0 1 exchange loc -->  1 0
pre_A = A.copy()
mutate(None, [A], 0.9, iris, [1, 2, 3, 4])

# 0 mutate on -->  2
counter = 1
titles = ['个体A基因1（变异前）', '个体A基因2（变异前）', '个体A基因3（变异前）', '个体A基因1（变异后）', '个体A基因2（变异后）', '个体A基因3（变异后）']
plt.figure(figsize=(15, 8))
for e in pre_A + A:
    plt.subplot(2, 3, counter)
    plot_tree(e, title=titles[counter - 1], node_size=500, font_size=10)
    counter += 1
plt.show()

# 读入基础数据
cemht = pd.read_csv(CEMHT)
X = cemht.drop(columns=['No', 'Y'])
Y = cemht.Y

# 对 X1~X4 进行标准化处理
X = X.apply(lambda __x: (__x - np.mean(__x)) / np.std(__x), axis=1)
X.head()
#          X1         X2          X3          X4
# 0  -0.812132   0.057192    -0.857886   1.612825
# 1  -1.234525   0.252215    -0.491155   1.473466
# 2  -0.666283   1.685303    -0.823055   -0.195965
# 3  -0.836854   0.426322    -1.026330   1.436862
# 4  -0.910705   1.431108    -0.962745   0.442343


std_error = evaluation_regression(X, Y)
print(std_error)

# 产生初始种群，假设种群规模为100
popSize = 100

# 设置特征长度为3
need_gs = 3

# 交叉重组触发概率
cross_prob = 0.85

# 突变概率
mutate_prob = 0.1

# 原始特征序号
feature_idx = [1, 2, 3, 4]

# 产生初始种群
individuals = gen_individuals(popSize, need_gs, X, feature_idx)
adjusts = []

for df in individuals['df']:
    adjusts.append(get_adjust(std_error, Y, df, evaluation_regression))

print(adjusts)

max_epochs = 100

for k in range(max_epochs):
    # 0.备份父代个体
    pre_individuals = copy.deepcopy(individuals)
    pre_adjusts = adjusts.copy()

    # 1.交叉
    inter_cross(individuals['df'], individuals['gene'], cross_prob)

    # 2.变异
    mutate(individuals['df'], individuals['gene'], mutate_prob, X, feature_idx)

    # 3.计算适应度
    adjusts = []
    for df in individuals['df']:
        adjusts.append(get_adjust(std_error, Y, df, evaluation_regression))

    # 4.合并，并按adjusts降序排列，取前0.4popSize个个体进行返回，对剩余的个体随机选取0.6popSize个返回
    pre_gene_keys = [''.join(e) for e in pre_individuals['gene']]
    gene_keys = [''.join(e) for e in individuals['gene']]

    for i in range(len(pre_gene_keys)):
        key = pre_gene_keys[i]
        if key not in gene_keys:
            individuals['df'].append(pre_individuals['df'][i])
            individuals['gene'].append(pre_individuals['gene'][i])
            adjusts.append(pre_adjusts[i])

    split_val = pd.Series(adjusts).quantile(q=0.6)
    index = list(range(len(adjusts)))
    need_delete_count = len(adjusts) - popSize
    random.shuffle(index)
    indices = []
    for i in index:
        if need_delete_count > 0:
            if adjusts[i] <= split_val:
                indices.append(i)
                need_delete_count = need_delete_count - 1
        else:
            break

    individuals['df'] = [i for j, i in enumerate(individuals['df']) if j not in indices]
    individuals['gene'] = [i for j, i in enumerate(individuals['gene']) if j not in indices]
    adjusts = [i for j, i in enumerate(adjusts) if j not in indices]
    alpha = np.max(adjusts) / np.mean(adjusts)
    if k % 100 == 99 or k == 0:
        print("第 ", k + 1, " 次迭代，最大适应度为 ", np.max(adjusts), " alpha : ", alpha)
    if np.mean(adjusts) > 0 and alpha < 1.001:
        print("进化终止，算法已收敛！ 共进化 ", k, " 代！")
        break


# 提取适应度最高的一个个体，获取其特征
loc = np.argmax(adjusts)
new_x = individuals['df'][loc]
print(new_x.head())

counter = 1
titles = ['特征-g1', '特征-g2', '特征-g3']
plt.figure(figsize=(10, 20))
for e in individuals['gene'][loc]:
    plt.subplot(3, 1, counter)
    plot_tree(e, title=titles[counter - 1], node_size=1000, font_size=13)
    counter = counter + 1
plt.show()
