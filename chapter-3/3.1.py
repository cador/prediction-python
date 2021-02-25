from utils import *

air_miles = pd.read_csv(AIR_MILES)
plot_acf(air_miles.miles, lags=10, title="air_miles 自相关性")
plt.show()

plot_pacf(air_miles.miles, lags=10, title="air_miles 偏相关性")
plt.show()

iris = pd.read_csv(IRIS)

# 参数说明
#     figsize=(10,10)，设置画布大小为10x10
#     alpha=1，设置透明度，此处设置为不透明
#     hist_kwds={"bins":20} 设置对角线上直方图参数
#     可通过设置diagonal参数为kde将对角图像设置为密度图
pd.plotting.scatter_matrix(iris, figsize=(10, 10), alpha=1, hist_kwds={"bins": 20})
plt.show()

sns.pairplot(iris, hue="Species")
plt.show()

# 重置变量名称
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_df = iris.drop(columns='Species')
iris_df.columns = features

# 此处，我们建立两个新变量，都存储花色分类值，其中type对应真实类别，cluster对应预测类别
iris_df['type'] = iris.Species
iris_df['cluster'] = iris.Species

# 将cluster变量转化为整数编码
iris_df.cluster = iris_df.cluster.astype('category')
iris_df.cluster = iris_df.cluster.cat.codes

# 将type变量转化为整数编码
iris_df.type = iris_df.type.astype('category')
iris_df.type = iris_df.type.cat.codes

# 获得花色类别列表
types = iris.Species.value_counts().index.tolist()

pair_plot(df=iris_df,
          plot_vars=features,
          colors=['#50B131', '#F77189', '#3BA3EC'],  # 指定描述三种花对应的颜色
          target_types=types,
          markers=['*', 'o', '^'],  # 指定预测类别cluster对应的形状
          color_col='type',  # 对应真实类别变量
          marker_col='cluster')  # 对应预测类别变量

dims = {'x': 'Sepal.Length', 'y': 'Petal.Length', 'z': 'Petal.Width'}
types = iris.Species.value_counts().index.tolist()

# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
for iris_type in types:
    tmp_data = iris[iris.Species == iris_type]
    x, y, z = tmp_data[dims['x']], tmp_data[dims['x']], tmp_data[dims['z']]
    ax.scatter(x, y, z, label=iris_type)

# 绘制图例
ax.legend(loc='upper left')

# 添加坐标轴（顺序是Z, Y, X）
ax.set_zlabel(dims['z'])
ax.set_ylabel(dims['y'])
ax.set_xlabel(dims['x'])
# 替换 plt.show()
plt.savefig('../tmp/三维散点图.png', bbox_inches='tight')

# 相关图 - 相关矩阵图

df = iris.drop(columns='Species')
corr = df.corr()
print(corr)

plt.close()
corr_plot(corr, c_map="Spectral", s=2000)

# 相关图 - 相关层次图
mtCars = pd.read_csv(MT_CARS)
mtCars.drop(columns="_", inplace=True)

# 计算第4种相异性度量
distance = np.sqrt(1 - mtCars.corr() * mtCars.corr())
row_clusters = linkage(pdist(distance, metric='euclidean'), method='ward')
dendrogram(row_clusters, labels=distance.index)
plt.tight_layout()
plt.ylabel('欧氏距离')
plt.plot([0, 2000], [1.5, 1.5], c='gray', linestyle='--')
plt.show()

# 互相关分析
lakeHuron = pd.read_csv(LAKE_HURON)
lhData = lakeHuron.query("1937 <= year <= 1960")
X, Y = air_miles.miles, lhData.level
out = ccf(X, Y, lag_max=10)
for i in range(len(out)):
    plt.plot([i, i], [0, out[i]], 'k-')
    plt.plot(i, out[i], 'ko')

plt.xlabel("lag", fontsize=14)
plt.xticks(range(21), range(-10, 11, 1))
plt.ylabel("ccf", fontsize=14)
plt.show()

# 典型相关分析
# 现以 iris 为例，说明 特征值与特征向量的计算过程
corr = iris.corr()

# 将corr进行分块，1:2两个变量一组，3:4是另外一组，并进行两两组合
X11 = corr.iloc[0:2, 0:2].values
X12 = corr.iloc[0:2, 2:4].values
X21 = corr.iloc[2:4, 0:2].values
X22 = corr.iloc[2:4, 2:4].values

# 按公式求解矩阵A和B
A = np.matmul(np.matmul(np.matmul(np.linalg.inv(X11), X12), np.linalg.inv(X22)), X21)
B = np.matmul(np.matmul(np.matmul(np.linalg.inv(X22), X21), np.linalg.inv(X11)), X12)

# 求解典型相关系数
A_eig_values, A_eig_vectors = np.linalg.eig(A)
B_eig_values, B_eig_vectors = np.linalg.eig(B)
print(np.sqrt(A_eig_values))
# array([0.940969  , 0.12393688])

# 下面验证 A 与 X^X_(-1) 、 B 与 Y^Y_(-1) 是否相等
# 进行验证
# 比较A与XΛX^(-1)是否相等
print(np.round(A - np.matmul(np.matmul(A_eig_vectors, np.diag(A_eig_values)), np.linalg.inv(A_eig_vectors)), 5))
#              Petal.Length      Petal.Width
# Sepal.Length        0.0         0.0
# Sepal.Width          0.0        -0.0

# 比较B与YΛY^(-1)是否相等
print(np.round(B - np.matmul(np.matmul(B_eig_vectors, np.diag(B_eig_values)), np.linalg.inv(B_eig_vectors)), 5))
#               Sepal.Length    Sepal.Width
# Petal.Length         0.0           0.0
# Petal.Width           0.0           0.0

# 将变量分组，并进行标准化处理
iris_g1 = iris.iloc[:, 0:2]
iris_g1 = iris_g1.apply(lambda __x: (__x - np.mean(__x)) / np.std(__x))
iris_g2 = iris.iloc[:, 2:4]
iris_g2 = iris_g2.apply(lambda __x: (__x - np.mean(__x)) / np.std(__x))

# 求解A对应的特征向量并计算典型向量C1
C1 = np.matmul(iris_g1, A_eig_vectors)
# 验证C1对应各变量的标准差是否为1，同时查看均值
print(C1.apply(np.std))
#  Sepal.Length    1.041196
#  Sepal.Width     0.951045
#  dtype: float64

print(C1.apply(np.mean))
# Sepal.Length   -1.894781e-16
# Sepal.Width    -9.000208e-16
# dtype: float64

# 由于均值为0，标准差不为1，这里对特征向量进行伸缩变换
eA = np.matmul(A_eig_vectors, np.diag(1 / C1.apply(np.std)))

# 再次验证方差和均值
C1 = np.matmul(iris_g1, A_eig_vectors)
print(C1.apply(np.std))
# Sepal.Length    1.0
# Sepal.Width     1.0
# dtype: float64

print(C1.apply(np.mean))
# Sepal.Length   -1.894781e-16
# Sepal.Width    -9.000208e-16
# dtype: float64

# 可见，特征向量已经满足要求，同理对B可得
C2 = np.matmul(iris_g2, B_eig_vectors)
print(C2.apply(np.std))
# Petal.Length    0.629124
# Petal.Width     0.200353
# dtype: float64

print(C2.apply(np.mean))
# Petal.Length   -1.421085e-16
# Petal.Width    -7.993606e-17
# dtype: float64

eB = np.matmul(B_eig_vectors, np.diag(1 / C2.apply(np.std)))
C2 = np.matmul(iris_g2, eB)
print(C2.apply(np.std))
# Petal.Length    1.0
# Petal.Width     1.0
# dtype: float64

print(C2.apply(np.mean))
# Petal.Length   -2.842171e-16
# Petal.Width    -4.144833e-16
# dtype: float64

# 所以，求得的特征值和特征向量分别为eV、eA、eB。进一步对C1、C2的相关性进行验证代码如下：
print(round(pd.concat([C1, C2], axis=1).corr(), 5))
#                   Sepal.Length    Sepal.Width     Petal.Length    3 Petal.Width
# Sepal.Length   1.00000     -0.00000             0.94097             -0.00000
# Sepal.Width     -0.00000     1.00000             0.00000             0.12394
# Petal.Length    0.94097      0.00000             1.00000             0.00000
# Petal.Width     -0.00000      0.12394             0.00000             1.00000

# 我们可以使用该模块直接求解两组数据的典型相关系数，现基于iris_g1与iris_g2，实现代码如下：

cca = CCA(n_components=2)
cca.fit(iris_g1, iris_g2)

# X_c与Y_c分别为转换之后的典型变量
X_c, Y_c = cca.transform(iris_g1, iris_g2)
print(round(pd.concat([pd.DataFrame(X_c, columns=iris_g1.columns),
                       pd.DataFrame(Y_c, columns=iris_g2.columns)], axis=1).corr(), 5))
#                    Sepal.Length   Sepal.Width Petal.Length    Petal.Width
# Sepal.Length   1.00000      0.00000         0.94097             -0.00000
# Sepal.Width     0.00000      1.00000         -0.00001             0.12394
# Petal.Length    0.94097      -0.00001       1.00000             -0.00000
# Petal.Width      -0.00000     0.12394        -0.00000            1.00000
