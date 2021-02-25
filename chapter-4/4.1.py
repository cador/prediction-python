from utils import *

boys = pd.read_csv(BOYS)
# 这里根据BMI指数，将bmi属性泛化成体重类型字段 w_type
parts = np.array([0, 18.5, 25, 28, 100])
types = ["过轻", "正常", "过重", "肥胖"]
boys['w_type'] = [types[np.where(x <= parts)[0][0] - 1] if not np.isnan(x) else "未知" for x in boys.bmi]
boys[boys.w_type != "未知"].w_type.value_counts().plot.bar()
plt.show()

# 按均匀分布生成 100 个介于 10 到 100 之间的实数
tmp = pd.Series([random.uniform(10, 100) for x in range(100)])

# 1.使用pd.Series下的 quantile 函数进行等比分箱，此处将数据分成4份
x = tmp.quantile(q=[0.25, 0.5, 0.75, 1])
x.index = ['A', 'B', 'C', 'D']
tmp_qtl = tmp.apply(lambda m: x[x >= m].index[0]).values
print(tmp_qtl)

# ...另外常可通过均值、中位数、最大/最小值来平滑数值以生成新的特征，这里用均值来举例
y = tmp.groupby(tmp_qtl).mean()
tmp_qtl_mean = [y[x] for x in tmp_qtl]
print(tmp_qtl_mean)

# 2.使用cut函数进行等宽分箱，此处将数据分成5份
tmp_cut = pd.cut(tmp, bins=5, labels=["B1", "B2", "B3", "B4", "B5"])

# ...另外可通过设置 labels 为NULL，并通过 levels 函数查看cut的水平
# ...进一步确定各分箱的取值区间
# ...可通过均值、中位数、最大/最小值来平滑数值以生成新的特征，这里拿均值来举例
z = tmp.groupby(tmp_cut).mean()
tmp_cut_mean = [z[x] for x in tmp_cut]
print(tmp_cut_mean)

iris = pd.read_csv(IRIS)
out = get_split_value(iris.Species, iris['Sepal.Length'].values)
print(out)
# 5.5
