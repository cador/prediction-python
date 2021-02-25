from utils import *

data = web.DataReader('600519.ss', 'yahoo', datetime.datetime(2014, 1, 1), datetime.datetime(2019, 9, 30))
seq_len = 21
dim_in = 4
dim_out = 4
pred_len = 30
v_mean = data.iloc[:, :4].apply(lambda x: np.mean(x))
v_std = data.iloc[:, :4].apply(lambda x: np.std(x))
t0 = data.iloc[:, :4].apply(lambda x: (x - np.mean(x)) / np.std(x)).values
X_train = np.zeros((t0.shape[0] - seq_len - pred_len, seq_len, dim_in))
Y_train = np.zeros((t0.shape[0] - seq_len - pred_len, dim_out), )
X_test = np.zeros((pred_len, seq_len, dim_in))
Y_test = np.zeros((pred_len, dim_out), )

for i in range(seq_len, t0.shape[0] - pred_len):
    Y_train[i - seq_len] = t0[i]
    X_train[i - seq_len] = t0[(i - seq_len):i]

for i in range(t0.shape[0] - pred_len, t0.shape[0]):
    Y_test[i - t0.shape[0] + pred_len] = t0[i]
    X_test[i - t0.shape[0] + pred_len] = t0[(i - seq_len):i]

model = Sequential()
model.add(LSTM(64, input_shape=(seq_len, dim_in), activation='relu', recurrent_dropout=0.01))
model.add(Dense(dim_out, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
history = model.fit(X_train, Y_train, epochs=200, batch_size=10, validation_split=0)
# Epoch 1/200
# 1350/1350 [==============================] - 1s 1ms/step - loss: 0.0447
# Epoch 2/200
# 1350/1350 [==============================] - 1s 737us/step - loss: 0.0059
# Epoch 3/200
# 1350/1350 [==============================] - 1s 743us/step - loss: 0.0043
# ......
# Epoch 200/200
# 1350/1350 [==============================] - 1s 821us/step - loss: 9.2794e-04

pred_df = model.predict(X_test) * v_std.values + v_mean.values

print(pred_df)
# array([[1069.35781887, 1038.57915742, 1056.77147186, 1053.83827734],
#       [1070.65142282, 1039.58533719, 1057.34561875, 1054.85567074],
#       [1083.58529328, 1052.70457308, 1070.78824637, 1067.49741882],
#
#       [1186.19297789, 1161.52758381, 1172.33666591, 1170.44623263],
#       [1181.42680223, 1155.14778501, 1166.5726204 , 1165.00336968],
#       [1186.75600881, 1160.84733425, 1172.37636963, 1170.09819923]])

print(pred_df.shape)
# (30, 4)


plt.figure(figsize=(10, 7))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(range(30), data.iloc[-30:data.shape[0], i].values, 'o-', c='black')
    plt.plot(range(30), pred_df[:, i], 'o--', c='gray')
    plt.ylim(1000, 1200)
    plt.ylabel("$" + data.columns[i] + "$")
plt.show()
v = 100 * (1 - np.sum(np.abs(pred_df - data.iloc[-30:data.shape[0], :4]).values) / np.sum(
    data.iloc[-30:data.shape[0], : 4].values))
print("Evaluation on test data: accuracy = %0.2f%% \n" % v)
# Evaluation on test data: accuracy = 99.01%
