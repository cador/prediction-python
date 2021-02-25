from utils import *

data = pd.read_csv(ENERGY_OUT)
missingno.matrix(data, labels=True, figsize=(45, 10))
plt.savefig('../tmp/missing_no1.png', bbox_inches='tight')

weather = pd.read_csv(WEATHER)

# 获取星期数据
data['weekday'] = [datetime.datetime.strptime(x, '%Y/%m/%d').weekday() for x in data.LOAD_DATE]

# 获取月份数据
data['month'] = [datetime.datetime.strptime(x, '%Y/%m/%d').month for x in data.LOAD_DATE]
data['date'] = [datetime.datetime.strptime(x, '%Y/%m/%d') for x in data.LOAD_DATE]

# 将数据按日期升序排列
data = data.sort_values(by='date')

# 获取时间趋势数据
data['trend'] = range(len(data))

# 设置索引并按索引进行关联
data = data.set_index('LOAD_DATE')
weather = weather.set_index('WETH_DATE')
p = data.join(weather)
p = p.drop(columns='date')

# 声明列表用于存储位置及插补值信息
out = list()
for index in np.where(p.apply(lambda x: np.sum(np.isnan(x)), axis=1) > 0)[0]:
    sel_col = np.logical_not(np.isnan(p.iloc[index]))
    use_col = np.where(sel_col)[0]
    cols = np.where(~sel_col)[0]
    for col in cols:
        nbs = np.where(p.iloc[:, use_col].apply(lambda x: np.sum(np.isnan(x)), axis=1) == 0)[0]
        nbs = nbs[nbs != index]
        nbs = (list(set(nbs).intersection(set(np.where(np.logical_not(np.isnan(p.iloc[:, col])))[0]))))
        t0 = [np.sqrt(np.sum((p.iloc[index, use_col] - p.iloc[x, use_col]) ** 2)) for x in nbs]
        t1 = 1 / np.array(t0)
        t_wts = t1 / np.sum(t1)
        out.append((index, col, np.sum(p.iloc[nbs, col].values * t_wts)))

print(out)
# [(18, 0, 15.862533989395969),
#  (18, 1, 15.066161287502018),
#  (18, 2, 14.19980745932543),
#  (18, 3, 13.58345782262413),
#  (18, 4, 13.51538093937498),
#  ...

for v in out:
    p.iloc[v[0], v[1]] = v[2]

print(p.head())
p.to_csv("../tmp/p_data.csv")

missingno.matrix(p, labels=True, figsize=(45, 10))
plt.savefig('../tmp/missing_no2.png', bbox_inches='tight')

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
plot_pacf(p.C010, ax=ax0, title="Partial Auto correlation of C010")
plot_pacf(p.C040, ax=ax1, title="Partial Auto correlation of C040")
plot_pacf(p.C060, ax=ax2, title="Partial Auto correlation of C060")
plot_pacf(p.C080, ax=ax3, title="Partial Auto correlation of C080")
plt.savefig('../tmp/partial_auto_cor.png', bbox_inches='tight')

cols = p.columns[[x.startswith('C') for x in p.columns]]
temps = ['MEAN_TMP', 'MIN_TMP', 'MAX_TMP']
t0 = pd.DataFrame([[p[t].corr(p[x]) for x in temps] for t in cols])
t0.columns = temps
t0.index = cols
plt.figure(figsize=(5, 10))
sns.heatmap(t0, linewidths=0.05, vmax=1, vmin=0, cmap='rainbow')
plt.show()

plt.figure(figsize=(10, 5))


def scale(x):
    return (x - np.mean(x)) / np.std(x)


plt.plot(range(p.shape[0]), scale(p.C010.values), 'o-', c='black', label="C010")
plt.plot(range(p.shape[0]), scale(p.MEAN_TMP.values), 'o--', c='gray', label="Mean Temperature")
plt.legend()
plt.ylabel("$scale \\quad value$")
plt.xlabel("$samples$")
plt.show()

t0 = p.groupby('weekday').mean()[['C040', 'C060']]
plt.plot(range(7), t0.C040, 'o--', c='black', label="C040")
plt.plot(range(7), t0.C060, 'o--', c='gray', label="C060")
plt.ylim(0, 150)
plt.xlabel("$weekday$")
plt.ylabel("$power\\quad load$")
plt.legend()
plt.show()
