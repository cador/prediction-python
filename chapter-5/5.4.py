import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = Axes3D(fig)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$y$', fontsize=16)
ax.set_zlabel('$z$', fontsize=16)
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)

# x-y 平面的网格
x, y = np.meshgrid(x, y)
z = x * np.exp(-x ** 2 - y ** 2)
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('cool'))
plt.savefig('../tmp/函数曲面.png', bbox_inches='tight')

# 初始化粒子群（包含20个粒子）
v_max = 1

# 设置惯性权重
w = 0.5

# 设置加速度常数
c1, c2 = 2, 2

# 设置最大迭代次数
iter_size = 100000

# 设置最佳适应度值的增量阈值
alpha = 0.0000001

# 在给定定义域内，随机生成位置矩阵如下
x_mat = np.random.uniform(-2, 2, (20, 2))

# 在给定最大速度的限制下，随机生成速度矩阵如下
v_mat = np.random.uniform(-v_max, v_max, (20, 2))


def get_adjust(location):
    """计算种群中所有粒子的适应度值"""
    __x, __y = location
    return -1*__x * np.exp(-1*__x ** 2 - __y ** 2)


adjusts = np.array([get_adjust(loc) for loc in x_mat])

p_best = x_mat, adjusts
g_best = x_mat[np.argmax(adjusts)], np.max(adjusts)
g_best_add = None

# 更新p_best、g_best，同时更新所有粒子的位置与速度
for k in range(iter_size):
    # 更新p_best，遍历adjusts，如果对应粒子的适应度是历史中最高的，则完成替换
    index = np.where(adjusts > p_best[1])[0]
    if len(index) > 0:
        p_best[0][index] = x_mat[index]
        p_best[1][index] = adjusts[index]

    # 更新g_best
    if np.sum(p_best[1] > g_best[1]) > 0:
        g_best_add = np.max(adjusts) - g_best[1]
        g_best = x_mat[np.argmax(adjusts)], np.max(adjusts)

    # 更新所有粒子的位置与速度
    x_mat_backup = x_mat.copy()
    x_mat = x_mat + v_mat
    v_mat = w * v_mat + c1 * np.random.uniform(0, 1) * (p_best[0] - x_mat_backup) + \
            c2 * np.random.uniform(0, 1) * (g_best[0] - x_mat_backup)

    # 如果v_mat有值超过了边界值，则设定为边界值
    x_mat[x_mat > 2] = 2
    x_mat[x_mat < (-2)] = -2
    v_mat[v_mat > v_max] = v_max
    v_mat[v_mat < (-v_max)] = -v_max

    # 计算更新后种群中所有粒子的适应度值
    adjusts = np.array([get_adjust(loc) for loc in x_mat])

    # 检查全局适应度值的增量，如果小于 alpha，则算法停止
    if g_best_add is not None and g_best_add < alpha:
        print("k = ", k, " 算法收敛！")
        break

print(g_best)
# (array([-0.70669195,  0.00303178]), -0.42887785271504353)
