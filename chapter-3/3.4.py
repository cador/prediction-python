from utils import *

# 将AirPassengers数据转换成环比值
ap = pd.read_csv(AIR_PASSENGERS)
ap['月份'] = ap.apply(lambda x: str(x['year']) + '-' + str(x['month']).rjust(2, '0'), axis=1)
ap_rows = len(ap)
ap_chain = ap.passengers[1:].values / ap.passengers[:(ap_rows - 1)].values
plt.figure(figsize=(15, 3))
plt.plot(range(ap_rows-1), ap_chain, 'ko-')
plt.plot([0, ap_rows-2], [1, 1], '--', c='gray')
x_index = np.arange(1, ap_rows, 20)
plt.xticks(x_index-1, ap.loc[x_index, '月份'], rotation=45)
plt.xlabel("月份")
plt.ylabel("乘客数量环比值")
plt.show()

# 环比值，离散化
ap_chain_lab = pd.cut(ap_chain, bins=4, include_lowest=True, labels=["A", "B", "C", "D"])
tmp_array = np.array([ap_chain_lab[i:(i + 10)] for i in range(len(ap_chain_lab) - 10 + 1)])

# 构建数据集
win_size = 10
con_df = pd.DataFrame(tmp_array, columns=["X%s" % x for x in range(1, win_size + 1)])

# 由于数据都是按时间先后顺序整理的，因此可用前80%提取规则，用后20%验证规则
con_train = con_df.loc[:int(len(con_df) * 0.8), :]
con_test = con_df.loc[len(con_train):len(con_df), :]

transactions = con_train.apply(lambda x: (x.index + '=' + x.values).tolist(), axis=1).values
association_rules = apriori(transactions, min_support=0.1, min_confidence=0.5, min_lift=1, min_length=2)
arules_out = arules_parse(list(association_rules))
print(arules_out['rules'].sort_values('conf', ascending=False).head())

tmp = con_test.query("X4=='B' and X9=='A'")
print("%d%%" % (100 * sum(tmp["X10"] == 'A') / tmp.shape[0]))
# 100%

# 声明变量，tuple_list用于存放所有的边，nodes_color、nodes_size分别存放节点的颜色和大小
# edges_size存放边的大小
tuple_list, nodes_color, nodes_size, edges_size = [], {}, {}, {}


# 自定义行处理函数
def row_proc(row):
    tmp_edges = []
    [tmp_edges.append((x, str(row.name))) for x in row['left'].split(",")]
    [tmp_edges.append((str(row.name), x)) for x in row['right'].split(",")]
    for e in row['left'].split(",") + row['right'].split(","):
        if e not in nodes_color:
            nodes_color[e] = 0
            nodes_size[e] = 600
    # 使用提升度来表示节点的颜色，颜色越深，提升度越大
    nodes_color[str(row.name)] = row['lift']
    # 使用置信度来表示节点的大小，节点越大，置信度也就越大
    nodes_size[str(row.name)] = 2 ** (row['conf'] * 10) * 3
    # 使用边的大小来表示规则的支持度，边越粗，支持度越大
    for k in tmp_edges:
        edges_size[k] = row['support'] * 20
        tuple_list.extend(tmp_edges)


arules_out['rules'].apply(row_proc, axis=1)
plt.figure(figsize=(10, 10))
# 建立有向图
G = nx.DiGraph()
G.add_edges_from(tuple_list)
pos = nx.kamada_kawai_layout(G)
colors = [nodes_color.get(node) for node in G.nodes()]
sizes = [nodes_size.get(node) for node in G.nodes()]
widths = [edges_size.get(edge) for edge in G.edges()]
nx.draw(G, pos, cmap=plt.get_cmap('Greys'), with_labels=True, width=widths,
        node_color=colors, node_size=sizes, edge_color='lightgray', font_color="lightgray")
plt.savefig('../tmp/网络图.png', bbox_inches='tight')

# 根据示例数据，手动输入构建包含事务信息的list对象
transactions = [['A', 'D'], ['A', 'E', 'F'], ['A', 'B', 'C', 'E', 'F'],
                ['B', 'C', 'D'], ['A', 'C', 'D', 'E', 'F'], ['A', 'B', 'D', 'F']]
rules = eclat(tracts=transactions, zmin=1, supp=50)
plt.close()
plt.figure(figsize=(5, 3))
tmp = pd.DataFrame(rules).sort_values(1, ascending=True)
tmp = tmp.set_index(tmp[0])
tmp[1].plot(kind='barh', color='gray')
plt.ylabel("频繁项集")
plt.xlabel("频率")
plt.show()

"""
:: 安装R包 => Python代码

from rpy2.robjects.packages import importr
utils = importr('utils')
utils.chooseCRANmirror(ind=20)
utils.install_packages('arulesSequences')

"""
vas = importr("arulesSequences")
arules = importr("arules")
s0 = vas.cspade(vas.read_baskets(
    con=ZAKI,
    info=StrVector(["sequenceID", "eventID", "SIZE"])),
    parameter=ListVector({"support": 0.5}))
arules.write(s0, '../tmp/zaki_out.txt')
print(pd.read_table('../tmp/zaki_out.txt', sep=" "))
