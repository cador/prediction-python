import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
import keras

data = pd.read_csv('../tmp/p_data.csv')
tmp = data.set_index('LOAD_DATE')
t0_min = tmp.apply(lambda x: np.min(x), axis=0).values
t0_ptp = tmp.apply(lambda x: np.ptp(x), axis=0).values
t0 = tmp.apply(lambda x: (x - np.min(x)) / np.ptp(x), axis=0).values

seq_len = 14
dim_in = 108
dim_out = 96
pred_len = 30
X_train = np.zeros((t0.shape[0] - seq_len - pred_len, seq_len, dim_in))
Y_train = np.zeros((t0.shape[0] - seq_len - pred_len, dim_out), )
X_test = np.zeros((pred_len, seq_len, dim_in))
Y_test = np.zeros((pred_len, dim_out), )
for i in range(seq_len, t0.shape[0] - pred_len):
    Y_train[i - seq_len] = t0[i][0:96]
    X_train[i - seq_len] = np.c_[t0[(i - seq_len):i], t0[i + 1][96:].repeat(seq_len).reshape((6, seq_len)).T]
for i in range(t0.shape[0] - pred_len, t0.shape[0]):
    Y_test[i - t0.shape[0] + pred_len] = t0[i][0:96]
    if i == t0.shape[0] - 1:
        # 这里weekday、trend、month和气温数据做了近似处理，正式使用时，需要使用天气预报的数据
        X_test[i - t0.shape[0] + pred_len] = np.c_[t0[(i - seq_len):i], t0[i][96:].repeat(seq_len).reshape((6, seq_len)).T]
    else:
        X_test[i - t0.shape[0] + pred_len] = np.c_[t0[(i - seq_len):i], t0[i + 1][96:].repeat(seq_len).reshape((6, seq_len)).T]

model = Sequential()
init = keras.initializers.glorot_uniform(seed=90)
model.add(LSTM(128, input_shape=(seq_len, dim_in), activation='relu', kernel_initializer=init,
               recurrent_dropout=0.01))
model.add(Dense(dim_out, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')
history = model.fit(X_train, Y_train, epochs=2000, batch_size=7, validation_split=0)
# Epoch 1/2000
# 351/351 [==============================] - 1s 4ms/step - loss: 0.0549
# Epoch 2/2000
# 351/351 [==============================] - 1s 2ms/step - loss: 0.0336
# Epoch 3/2000
# 351/351 [==============================] - 1s 1ms/step - loss: 0.0293
# ...
# Epoch 2000/2000
# 351/351 [==============================] - 1s 2ms/step - loss: 6.9224e-04

pred_y = model.predict(X_test)
pred_df = pred_y * t0_ptp[0:96] + t0_min[0:96]

print(pred_df)
# array([[15.38415141, 16.00260305, 18.73421875, ..., 26.73801819,
#        21.77837609, 23.18869093],
#       [16.06377707, 16.4213548 , 20.20301449, ..., 21.35637292,
#        23.28361729, 23.69081441],
#       [16.99017648, 17.00997578, 19.17132048, ..., 23.23738699,
#        19.08706516, 16.54340419],
#       ...,
#       [15.82022316, 18.33014316, 17.44376823, ..., 23.52037691,
#        21.772848  , 17.89651412],
#       [15.07703572, 15.00410491, 19.98956045, ..., 24.61383956,
#        22.06024055, 19.6154434 ],
#       [13.23139953, 19.16844683, 17.36724287, ..., 22.46061762,
#        23.24737013, 22.35341081]])

print(pred_df.shape)
# (30, 96)

real_df = Y_test * t0_ptp[0:96] + t0_min[0:96]
base = 0
error = 0
plt.figure(figsize=(20, 10))
for index in range(0, 30):
    real_array = real_df[index][0:96]
    pred_array = real_df[index][0:96]
    pred_array[np.where(pred_array < 0)] = 0
    plt.subplot(5, 7, index + 1)
    plt.plot(range(96), real_array, '-', label="real", c='black')
    plt.plot(range(96), pred_array, '--', label="pred", c='gray')
    plt.ylim(0, 250)
    base = base + np.sum(real_array)
    error = error + np.sum(np.abs(real_array - pred_array))
plt.show()
v = 100 * (1 - error / base)
print("Evaluation on test data: accuracy = %0.2f%% \n" % v)
# Evaluation on test data: accuracy = 74.95%
