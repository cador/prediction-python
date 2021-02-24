from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.signal as sg
from sklearn import tree
from sklearn import linear_model
import networkx as nx
import random
import re
from sklearn.metrics import confusion_matrix

min_number = 0.01

# 定义二元运算函数的集合
two_group = ['add', 'sub', 'times', 'div']

# 定义一元运算函数的集合
one_group = ['log', 'sqrt', 'pow2', 'pow3', 'inv', 'sigmoid', 'tanh', 'relu', 'binary']


class Node:
    def __init__(self, value, label, left=None, right=None):
        self.value = value
        self.label = label
        self.left = left
        self.right = right


def pair_plot(df, plot_vars, colors, target_types, markers, color_col, marker_col, fig_size=(15, 15)):
    # 设置画布大小
    plt.figure(figsize=fig_size)
    plot_len = len(plot_vars)
    index = 0
    for p_col in range(plot_len):
        col_index = 1
        for p_row in range(plot_len):
            index = index + 1
            plt.subplot(plot_len, plot_len, index)
            if p_row != p_col:
                # 非对角位置，绘制散点图
                df.apply(lambda row: plt.plot(row[plot_vars[p_row]], row[plot_vars[p_col]],
                                              color=colors[int(row[color_col])],
                                              marker=markers[int(row[marker_col])], linestyle=''), axis=1)
            else:
                # 对角位置，绘制密度图
                for ci in range(len(colors)):
                    sns.kdeplot(df.iloc[np.where(df[color_col] == ci)[0], p_row],
                                shade=True, color=colors[ci], label=target_types[ci])
            # 添加横纵坐标轴标签
            if col_index == 1:
                plt.ylabel(plot_vars[p_col])
                col_index = col_index + 1
            if p_col == plot_len - 1:
                plt.xlabel(plot_vars[p_row])
    plt.show()


def corr_plot(corr, c_map, s):
    # 使用x,y,z来存储变量对应矩阵中的位置信息，以及相关系数
    x, y, z = [], [], []
    __n__ = corr.shape[0]
    for row in range(__n__):
        for column in range(__n__):
            x.append(row)
            y.append(__n__ - 1 - column)
            z.append(round(corr.iloc[row, column], 2))
    # 使用scatter函数绘制圆圈矩阵
    sc = plt.scatter(x, y, c=z, vmin=-1, vmax=1, s=s * np.abs(z), cmap=plt.cm.get_cmap(c_map))
    # 添加颜色板
    plt.colorbar(sc)
    # 设置横纵坐标轴的区间范围
    plt.xlim((-0.5, __n__ - 0.5))
    plt.ylim((-0.5, __n__ - 0.5))
    # 设置横纵坐标轴值标签
    plt.xticks(range(__n__), corr.columns, rotation=90)
    plt.yticks(range(__n__)[::-1], corr.columns)
    # 去掉默认网格
    plt.grid(False)
    # 使用顶部的轴作为横轴
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    # 重新绘制网格线
    internal_space = [0.5 + k for k in range(4)]
    [plt.plot([m, m], [-0.5, __n__ - 0.5], c='lightgray') for m in internal_space]
    [plt.plot([-0.5, __n__ - 0.5], [m, m], c='lightgray') for m in internal_space]
    # 显示图形
    plt.show()


def ccf(x, y, lag_max=100):
    result = sg.correlate(y - np.mean(y), x - np.mean(x), method='direct') / \
             (np.std(y) * np.std(x) * len(x))
    length = int((len(result) - 1) / 2)
    low = length - lag_max
    high = length + (lag_max + 1)
    return result[low:high]


def arules_parse(association_results):
    """
    使用 Apriori 算法，提取关联规则
    :param association_results:
    :return:
    """
    freq_items = list()
    freq_items_support = list()
    left_items = list()
    right_items = list()
    conf = list()
    lift = list()
    rule_support = list()
    for item in association_results:
        freq_items.append(",".join(item[0]))
        freq_items_support.append(item[1])
        for e in item[2]:
            left_items.append(",".join(e[0]))
            right_items.append(",".join(e[1]))
            conf.append(e[2])
            lift.append(e[3])
            rule_support.append(item[1])
    return {
        "freq_items": pd.DataFrame({'items': freq_items, 'support': freq_items_support}),
        "rules": pd.DataFrame(
            {'left': left_items, 'right': right_items, 'support': rule_support, 'conf': conf, 'lift': lift})
    }


def std_proc(x, is_positive=True):
    """
    该函数用于获取数据x的各种标准化值
    is_positive:
    :param x: 用于标准化的实数数组
    :param is_positive: 是否是正向指标
    :return:
    """
    x = np.array(x)
    v_max, v_min, v_std, v_mean = np.max(x), np.min(x), np.std(x), np.mean(x)
    # 1. 线性标准化
    # --- 极差标准化
    if v_max > v_min:
        y_ext = (x - v_min if is_positive else v_max - x) / (v_max - v_min)
    else:
        print("最大值与最小值相等，不能进行极差标准化!")
        y_ext = None

    # --- z-score标准化
    if v_std == 0:
        print("由于标准差为0，不能进行z-score标准化")
        y_zsc = None
    else:
        y_zsc = (x - v_mean) / v_std

    # --- 小数定标标准化
    y_pot = x / (10 ** len(str(np.max(np.abs(x)))))

    # 2. 非线性标准化
    # --- 对数标准化
    y = np.log((x - v_min if is_positive else v_max - x) + 1)
    y_log = y / np.max(y)

    # --- 倒数标准化
    y_inv = np.min(np.abs(x[x != 0])) / x
    return {"y_ext": y_ext, "y_zsc": y_zsc, "y_pot": y_pot, "y_log": y_log, "y_inv": y_inv}


def gains(u, v):
    unique_list = [np.unique(u, return_counts=True)[1]]
    ent_u = [np.sum([p * np.log2(1 / p) for p in ct / np.sum(ct)]) for ct in unique_list][0]
    v_id, v_ct = np.unique(v, return_counts=True)
    ent_u_m = [np.sum([p * np.log2(1 / p) for p in ct / np.sum(ct)]) for ct in
               [np.unique(u[v == m], return_counts=True)[1] for m in v_id]]
    return ent_u - np.sum(np.array(ent_u_m) * (v_ct / np.sum(v_ct)))


def get_split_value(u, x):
    sorted_x, max_gains, e_split = np.msort(x), 0, min(x)
    for e in sorted_x:
        tmp = np.zeros(len(x))
        tmp[x > e] = 1
        tmp_gain = gains(u, tmp)
        if tmp_gain > max_gains:
            max_gains, e_split = tmp_gain, e
    return e_split


def disc(u, x):
    sorted_x = np.msort(x)
    max_gains, max_tmp = 0, None
    for e in sorted_x:
        tmp = np.zeros(len(x))
        tmp[x > e] = 1
        tmp_gain = gains(u, tmp)
        if tmp_gain > max_gains:
            max_gains, max_tmp = tmp_gain, tmp
    return max_tmp


def gains_ratio(u, v):
    unique_list = [np.unique(v, return_counts=True)[1]]
    ent_v = [np.sum([p * np.log2(1 / p) for p in ct / np.sum(ct)]) for ct in unique_list][0]
    return gains(u, v) / ent_v


def eval_func(x_train, y_train, x_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf_fit = clf.fit(x_train, y_train)
    c_mat = confusion_matrix(y_test, clf_fit.predict(x_test))
    return np.sum(np.diag(c_mat)) / x_test.shape[0]


def g(f, a, b=None):
    """
    f: 一元或二元运算函数
    a: 第一个参数
    b: 如果f是一元运算函数，则b为空，否则代表二元运算的第二个参数
    """
    if b is None:
        return f(a)
    else:
        return f(a, b)


# 一元运算
def log(x):
    return np.sign(x) * np.log2(np.abs(x) + 1)


def sqrt(x):
    return np.sqrt(x - np.min(x) + min_number)


def pow2(x):
    return x ** 2


def pow3(x):
    return x ** 3


def inv(x):
    return 1 * np.sign(x) / (np.abs(x) + min_number)


def sigmoid(x):
    if np.std(x) < min_number:
        return x
    x = (x - np.mean(x)) / np.std(x)
    return (1 + np.exp(-x)) ** (-1)


def tanh(x):
    if np.std(x) < min_number:
        return x
    x = (x - np.mean(x)) / np.std(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    if np.std(x) < min_number:
        return x
    x = (x - np.mean(x)) / np.std(x)
    return np.array([e if e > 0 else 0 for e in x])


def binary(x):
    if np.std(x) < min_number:
        return x
    x = (x - np.mean(x)) / np.std(x)
    return np.array([1 if e > 0 else 0 for e in x])


# 二元运算
def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def times(x, y):
    return x * y


def div(x, y):
    return x * np.sign(y) / (np.abs(y) + min_number)


# 随机增加一元运算符
def add_one_group(feature_string, prob=0.3):
    random_float = np.random.uniform(0, 1)
    return 'g({0},{1})'.format(random.choice(one_group), feature_string) if random_float < prob else feature_string


# 构建满二叉树，并生成数学表达式
def gen_full_tree_exp(var_flag_array):
    half_n = len(var_flag_array) // 2
    middle_array = []
    for i in range(half_n):
        if var_flag_array[i] == '0' and var_flag_array[i + half_n] != '0':
            middle_array.append('g({0},{1})'.format(random.choice(one_group),
                                                    add_one_group(var_flag_array[i + half_n])))
        elif var_flag_array[i] != '0' and var_flag_array[i + half_n] == '0':
            middle_array.append('g({0},{1})'.format(random.choice(one_group),
                                                    add_one_group(var_flag_array[i])))
        elif var_flag_array[i] != '0' and var_flag_array[i + half_n] != '0':
            middle_array.append('g({0},{1},{2})'.format(random.choice(two_group),
                                                        add_one_group(var_flag_array[i]),
                                                        add_one_group(var_flag_array[i + half_n])))
    if len(middle_array) == 1:
        return add_one_group(middle_array[0])
    else:
        return gen_full_tree_exp(middle_array)


# 构建偏二叉树，并生成数学表达式
def gen_side_tree_exp(var_flag_array):
    if len(var_flag_array) == 1:
        return add_one_group(var_flag_array[0])
    else:
        var_flag_array[1] = 'g({0},{1},{2})'.format(random.choice(two_group),
                                                    add_one_group(var_flag_array[0]),
                                                    add_one_group(var_flag_array[1]))
        del var_flag_array[0]
        return gen_side_tree_exp(var_flag_array)


def random_get_tree(input_data, feature_idx, n_max=10):
    """
    从原始数据特征中，随机获取特征表达式
    feature_idx: 原始特征的下标数值，最小从1开始
    n_max:一次最多从特征中可放回抽样次数，默认为10
    """
    data = pd.DataFrame({"X" + str(e): input_data.iloc[:, (e - 1)].values for e in feature_idx})
    # 随机抽取N个特征下标
    __n__ = random.choice(range(2, n_max + 1))

    # 随机决定是使用满二叉树还是偏二叉树
    if random.choice([0, 1]) == 1:
        # 选择满二叉树
        select_feature_index = [random.choice(feature_idx) for i in range(__n__)] + [0] * int(2 ** np.ceil(np.log2(__n__)) - __n__)
        random.shuffle(select_feature_index)
        select_feature_index = ['data.X' + str(e) + ".values" if e > 0 else '0' for e in select_feature_index]
        tree_exp = gen_full_tree_exp(select_feature_index)
    else:
        # 选择偏二叉树
        select_feature_index = ['data.X' + str(e) + ".values" for e in [random.choice(feature_idx) for i in range(__n__)]]
        tree_exp = gen_side_tree_exp(select_feature_index)

    return {"f_value": eval(tree_exp), "tree_exp": tree_exp.replace("data.", "").replace(".values", "")}


def transform(feature_string):
    my_dict = {}
    pattern = r'g\([^\(\)]*\)'
    so = re.search(pattern, feature_string)
    while so:
        start, end = so.span()
        key = len(my_dict)
        my_dict[key] = so.group()
        feature_string = feature_string[0:start] + '<' + str(key) + '>' + feature_string[end:]
        so = re.search(pattern, feature_string)
    return my_dict


def parse(group_unit):
    tmp = group_unit.lstrip("g(").rstrip(")").split(',')
    tmp = tmp + [None] if len(tmp) == 2 else tmp
    return [int(x[1:-1]) if x is not None and re.match(r'<[0-9]+>', x) else x for x in tmp]


def bi_tree(mapping, start_no, index=0, labels=None):
    if labels is None:
        labels = dict()
    name, left, right = parse(mapping[start_no])
    if left is not None:
        if type(left) == int:
            left_node, s_labels, max_index = bi_tree(mapping, left, index + 1, labels)
            labels = s_labels
        else:
            left_node = Node(index + 1, left)
            labels[index + 1] = left
            max_index = index + 1
    else:
        left_node = None
        max_index = 1

    if right is not None:
        if type(right) == int:
            right_node, s_labels, max_index = bi_tree(mapping, right, max_index + 1, labels)
            labels = s_labels
        else:
            right_node = Node(max_index + 1, right)
            labels[max_index + 1] = right
            max_index = max_index + 1
    else:
        right_node = None

    labels[index] = name
    return Node(index, name, left_node, right_node), labels, max_index


def create_graph(__g, node, pos=None, x=0.0, y=0, layer=1):
    if pos is None:
        pos = dict()
    pos[node.value] = (x, y)
    if node.left:
        __g.add_edge(node.value, node.left.value)
        l_x, l_y = x - 1 / layer, y - 1
        l_layer = layer + 1
        create_graph(__g, node.left, x=l_x, y=l_y, pos=pos, layer=l_layer)
    if node.right:
        __g.add_edge(node.value, node.right.value)
        r_x, r_y = x + 1 / layer, y - 1
        r_layer = layer + 1
        create_graph(__g, node.right, x=r_x, y=r_y, pos=pos, layer=r_layer)
    return __g, pos


def plot_tree(feature_string, title=None, node_size=5000, font_size=18):
    my_dict = transform(feature_string)
    root, labels, _ = bi_tree(my_dict, len(my_dict) - 1, 0, labels={})
    graph = nx.Graph()
    graph, pos = create_graph(graph, root)
    nx.draw_networkx(graph, pos, node_size=node_size, width=2, node_color='black', font_color='white',
                     font_size=font_size, with_labels=True, labels=labels)
    plt.axis('off')
    if title is not None:
        plt.title(title)


def gen_individuals(k, gen_num, input_data, feature_idx, n_max=10):
    """产生k个个体, gen_num表示每个体对应的固定基因数量"""
    individual_list = []
    gene_list = []
    for e in range(k):
        individual = {}
        gene = []
        for i in range(gen_num):
            out = random_get_tree(input_data, feature_idx, n_max)
            individual["g" + str(i + 1)] = out['f_value']
            gene.append(out['tree_exp'])
        individual = pd.DataFrame(individual)
        individual_list.append(individual)
        gene_list.append(gene)
    return {"df": individual_list, "gene": gene_list}


def get_adjust(std_error, y, individual_data, handle):
    """计算适应度，通过外部定义的handle来处理，同时适用于分类和回归问题"""
    cur_error = handle(individual_data, y)
    return std_error - cur_error if std_error > cur_error else 0


def evaluation_classify(x, y):
    """建立分类问题的评估方法"""
    clf = tree.DecisionTreeClassifier(random_state=0)
    errors = []
    for i in range(len(x)):
        index = [e for e in range(x.shape[0])]
        index.remove(i)
        x_train = x.iloc[index, :]
        x_test = x.iloc[[i], :]
        y_train = y[index]
        y_test = y[i]
        clf.fit(x_train, y_train)
        errors.extend([clf.predict(x_test) != y_test])
    return np.sum(errors) / len(errors)


def evaluation_regression(x, y):
    """建立回归问题的评估方法"""
    reg = linear_model.LinearRegression()
    errors = 0
    for i in range(len(x)):
        index = [e for e in range(len(x))]
        index.remove(i)
        x_train = x.iloc[index, :]
        x_test = x.iloc[[i], :]
        y_train = y[index]
        y_test = y[i]
        reg.fit(x_train, y_train)
        errors = errors + (y_test - reg.predict(x_test)[0]) ** 2
    return errors / np.sum(y)


def inter_cross(individual_list, gene_list, prob):
    """ 对染色体进行交叉操作 """
    gene_num = len(gene_list[0])
    ready_index = list(range(len(gene_list)))
    while len(ready_index) >= 2:
        d1 = random.choice(ready_index)
        ready_index.remove(d1)
        d2 = random.choice(ready_index)
        ready_index.remove(d2)

        if np.random.uniform(0, 1) <= prob:
            loc = random.choice(range(gene_num))
            print(d1, d2, "exchange loc --> ", loc)

            # 对数据做交叉操作
            if individual_list is not None:
                tmp = individual_list[d1].iloc[:, loc]
                individual_list[d1].iloc[:, loc] = individual_list[d2].iloc[:, loc]
                individual_list[d2].iloc[:, loc] = tmp

            # 对基因型做交叉操作
            tmp = gene_list[d1][loc]
            gene_list[d1][loc] = gene_list[d2][loc]
            gene_list[d2][loc] = tmp


def mutate(individual_list, gene_list, prob, input_data, feature_idx, n_max=10):
    gene_num = len(gene_list[0])
    ready_index = list(range(len(gene_list)))
    for i in ready_index:
        if np.random.uniform(0, 1) <= prob:
            loc = random.choice(range(gene_num))
            print(i, "mutate on --> ", loc)
            tmp = random_get_tree(input_data, feature_idx, n_max)
            if individual_list is not None:
                individual_list[i].iloc[:, loc] = tmp['f_value']
            gene_list[i][loc] = tmp['tree_exp']
