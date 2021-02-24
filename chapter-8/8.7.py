import pandas as pd
import numpy as np
from utils.data_path import CANADA
from keras.layers import LSTM, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

SEQLEN = 15
dim_in = 4
dim_out = 4

src_canada = pd.read_csv(CANADA)
tmp = src_canada.drop(columns=['year', 'season'])
v_mean = tmp.apply(lambda x: np.mean(x))
v_std = tmp.apply(lambda x: np.std(x))
t0 = tmp.apply(lambda x: (x - np.mean(x)) / np.std(x)).values
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
model.add(LSTM(45, input_shape=(SEQLEN, dim_in), activation='relu', recurrent_dropout=0.01))
model.add(Dense(dim_out, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
history = model.fit(X_train, Y_train, epochs=2000, batch_size=5, validation_split=0)
# Epoch 1/2000
# 61/61 [==============================] - 0s 6ms/step - loss: 0.4616
# Epoch 2/2000
# 61/61 [==============================] - 0s 886us/step - loss: 0.3107
# Epoch 3/2000
# 61/61 [==============================] - 0s 1ms/step - loss: 0.1842
# ......
# Epoch 2000/2000
# 61/61 [==============================] - 0s 768us/step - loss: 0.0011

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
