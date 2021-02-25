from utils import *


def kalman(Z, A=None, H=None, Q=None, R=None, X0=None, P0=None):
    """
        该函数对 Kalman 滤波算法进行实现
        :param Z:观测量
        :param A:状态转移矩阵，默认初始化为 np.identity(Z.shape[1])
        :param H:观测协方差矩阵，默认初始化为 np.identity(Z.shape[1])
        :param Q:系统噪声协方差矩阵，默认初始化为 np.identity(Z.shape[1])
        :param R:观测噪声协方差矩阵，默认初始化为 np.identity(Z.shape[1])
        :param X0:状态量初始值，默认初始化为 np.identity(Z.shape[1])
        :param P0:误差协方差矩阵，默认初始化为 np.identity(Z.shape[1])
    """
    dmt = np.identity(Z.shape[1])
    A, H, Q, R, X0, P0 = [e if e is not None else dmt for e in [A, H, Q, R, X0, P0]]
    X = [X0]
    P = [P0]
    N = Z.shape[0]
    I = np.identity(A.shape[0])
    for i in range(N):
        # 均方误差的一步预测方程
        Pp = np.matmul(np.matmul(A, P[i]), A.T) + Q
        # 滤波增益方程（权重）
        K = np.matmul(np.matmul(Pp, H.T), np.linalg.inv(np.matmul(np.matmul(H, Pp), H.T) + R))
        # 状态的一步预测方程
        Xp = np.matmul(A, X[i])
        # 滤波估计方程（k时刻的最优值）
        X.append(Xp + np.matmul(K, np.identity(Z.shape[1]) * Z[i, :] - np.matmul(H, Xp)))
        # 均方误差更新矩阵（k时刻的最优均方误差）
        P.append(np.matmul(I - np.matmul(K, H), Pp))
    return X


src_canada = pd.read_csv(CANADA)
val_columns = ['e', 'prod', 'rw', 'U']
Z = src_canada[val_columns].values
X = kalman(Z)
out = []
[out.append(np.diag(e)) for e in X[1::]]
out = np.array(out)

xts = src_canada[['year', 'season']].apply(lambda x: str(x[0]) + '-' + x[1], axis=1).values
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
index = 0
for ax in axes.flatten():
    ax.plot(range(out.shape[0]), src_canada[val_columns[index]], '-', c='lightgray', linewidth=2, label="real")
    ax.set_xticks(range(out.shape[0])[::10])
    ax.set_xticklabels(xts[::10], rotation=50)
    ax.plot(range(5, out.shape[0]), out[5:, index], '--', c='black', linewidth=2, label="predict")
    ax.set_ylabel("$" + src_canada.columns[index] + "$", fontsize=14)
    ax.legend()
    index = index + 1
plt.tight_layout()
plt.show()
