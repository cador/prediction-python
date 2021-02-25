from utils import *


def c(s):
    """自定义目标函数C"""
    return 1 / (s * np.sin(s) + 12)


# 初始化
# 设定初始温度
t0 = np.var(np.random.uniform(0, 12.55, 100))

# 设定初始解
s0 = np.random.uniform(0, 12.55, 1)

# 设定迭代次数
iter_size = 3000

# 设定终止条件，连续ct个新解都没有接受时终止算法
ct = 200
ct_array = []

# 保存历史最好的状态，默认取上边界值
best = 12.55

for t in range(1, iter_size + 1):
    # 在s0附近产生新解，但又能包含定义内的所有值
    s1 = np.random.normal(s0, 2, 1)
    while s1 < 0 or s1 > 12.55:
        s1 = np.random.normal(s0, 2, 1)

    # 计算能量增量
    delta_t = c(s1) - c(s0)
    if delta_t < 0:
        s0 = s1
        ct_array.append(1)
    else:
        p = np.exp(-delta_t / t0)
        if np.random.uniform(0, 1) < p:
            s0 = s1
            ct_array.append(1)
        else:
            ct_array.append(0)

    best = s0 if c(s0) < c(best) else best

    # 更新温度
    t0 = t0 / np.log(1 + t)

    # 检查终止条件
    if len(ct_array) > ct and np.sum(ct_array[-ct:]) == 0:
        print("迭代 ", t, " 次，连续 ", ct, " 次没有接受新解，算法终止！")
        break

# 状态最终停留位置
print(s0)
# array([7.98092592])

# 最佳状态，即对应最优解的状态
print(best)
# 迭代  363  次，连续  200  次没有接受新解，算法终止！
# array([7.98092592])
