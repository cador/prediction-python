from utils import *

src_canada = pd.read_csv(CANADA)
tmp = src_canada.drop(columns=['year', 'season'])

# 计算标准化操作对应的均值向量与标准差向量
v_mean = tmp.apply(lambda x: np.mean(x))
v_std = tmp.apply(lambda x: np.std(x))

# 对基础数据进行标准化处理
t0 = tmp.apply(lambda x: (x - np.mean(x)) / np.std(x)).values

# 定义输入序列长度、输入与输出的维度
SEQLEN = 6
dim_in = 4
dim_out = 4

# 定义训练集与测试集的基础数据，并完成构建。这里使用最后8条数据进行测试
X_train = np.zeros((t0.shape[0] - SEQLEN - 8, SEQLEN, dim_in))
Y_train = np.zeros((t0.shape[0] - SEQLEN - 8, dim_out), )
X_test = np.zeros((8, SEQLEN, dim_in))
Y_test = np.zeros((8, dim_out), )
for i in range(SEQLEN, t0.shape[0] - 8):
    Y_train[i - SEQLEN] = t0[i]
    X_train[i - SEQLEN] = t0[(i - SEQLEN):i]
for i in range(t0.shape[0] - 8, t0.shape[0]):
    Y_test[i - t0.shape[0] + 8] = t0[i]
    X_test[i - t0.shape[0] + 8] = t0[(i - SEQLEN):i]

model = Sequential()
model.add(SimpleRNN(128, input_shape=(SEQLEN, dim_in), activation='relu', recurrent_dropout=0.01))
model.add(Dense(dim_out, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
history = model.fit(X_train, Y_train, epochs=1000, batch_size=2, validation_split=0)
# Epoch 1/1000
# 70/70 [==============================] - 0s 6ms/step - loss: 0.2661
# Epoch 2/1000
# 70/70 [==============================] - 0s 729us/step - loss: 0.0568
# Epoch 3/1000
# 70/70 [==============================] - 0s 729us/step - loss: 0.0431
# ......
# Epoch 999/1000
# 70/70 [==============================] - 0s 972us/step - loss: 7.9727e-04
# Epoch 1000/1000
# 70/70 [==============================] - 0s 915us/step - loss: 7.6667e-04


pred_df = model.predict(X_test) * v_std.values + v_mean.values
m = 16
xts = src_canada[['year', 'season']].iloc[-m:].apply(lambda x: str(x[0]) + '-' + x[1], axis=1).values
cols = src_canada.drop(columns=['year', 'season']).columns
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
index = 0
for ax in axes.flatten():
    ax.plot(range(m), src_canada[cols].iloc[-m:, index], '-', c='lightgray', linewidth=2, label="real")
    ax.plot(range(m - 8, m), pred_df[:, index], 'o--', c='black', linewidth=2, label="predict")
    ax.set_xticks(range(m))
    ax.set_xticklabels(xts, rotation=50)
    ax.set_ylabel("$" + cols[index] + "$", fontsize=14)
    ax.legend()
    index = index + 1
plt.tight_layout()
plt.show()
