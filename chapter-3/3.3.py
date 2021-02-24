from sklearn.cluster import KMeans
from utils.data_path import AIR_PASSENGERS
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Mac 系统，使用如下设置可解决图表中文乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 加载数据、转换并进行标准化处理
passengers = pd.read_csv(AIR_PASSENGERS)
data = list()
tmp = passengers.groupby('year').filter(
    lambda block: data.append([block.iloc[0, 0]] + block.passengers.values.tolist())
)
data = pd.DataFrame(data)
data.set_index(data[0], inplace=True)
data.drop(columns=0, inplace=True)

# 标准化时，采取按行标准化的方法，即每行中都是0和1，分别表示最大和最小，以此方式来分析曲线模式
data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=1)

# 假如我要构造一个聚类数为2的聚类器
km_cluster = KMeans(n_clusters=2, max_iter=300, n_init=40, init='k-means++')
km_cluster.fit(data)
data['cluster'] = km_cluster.labels_

# 绘制聚类结果曲线（两个类别）
styles = ['co-', 'ro-']
data.apply(lambda row: plt.plot(np.arange(1, 13), row[np.arange(1, 13)], styles[int(row.cluster)], alpha=0.3), axis=1)
plt.xlabel("月份")
plt.ylabel("乘客数量(标准化)")
plt.show()

# 绘制聚类树状图，发现适合聚成两类，并添加辅助线标记
row_clusters = linkage(pdist(data, metric='euclidean'), method='ward')
dendrogram(row_clusters)
plt.tight_layout()
plt.ylabel('欧氏距离')
plt.axhline(0.6, c='red', linestyle='--')
plt.show()

ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
data['cluster'] = ac.fit_predict(data)
data.apply(lambda row: plt.plot(np.arange(1, 13), row[np.arange(1, 13)], styles[int(row.cluster)], alpha=0.3), axis=1)
plt.xlabel("月份")
plt.ylabel("乘客数量(标准化)")
plt.show()
