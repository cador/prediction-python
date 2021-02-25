from utils import *

data = pd.read_csv('../tmp/p_data.csv')
data = data.set_index('LOAD_DATE')
parts = 14
this_one = data.iloc[parts:]
bak_index = this_one.index
for k in range(1, parts + 1):
    last_one = data.iloc[(parts - k):(this_one.shape[0] - k + parts)]
    this_one.set_index(last_one.index, drop=True, inplace=True)
    this_one = this_one.join(last_one, lsuffix="", rsuffix="_p" + str(k))

this_one.set_index(bak_index, drop=True, inplace=True)
this_one = this_one.fillna(0)
t0 = this_one.iloc[:, 0:96]
t0_min = t0.apply(lambda x: np.min(x), axis=0).values
t0_ptp = t0.apply(lambda x: np.ptp(x), axis=0).values
this_one = this_one.apply(lambda x: (x - np.min(x)) / np.ptp(x), axis=0)

test_data = this_one.iloc[-30:]
train_data = this_one.iloc[:-30]
train_y_df = train_data.iloc[:, 0:96]
train_y = np.array(train_y_df)
train_x_df = train_data.iloc[:, 96:]
train_x = np.array(train_x_df)

test_y_df = test_data.iloc[:, 0:96]
test_y = np.array(test_y_df)
test_x_df = test_data.iloc[:, 96:]
test_x = np.array(test_x_df)
test_y_real = t0.iloc[-30:]

init = keras.initializers.glorot_uniform(seed=1)
simple_adam = keras.optimizers.Adam()
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=512, input_dim=1434, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dense(units=256, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dense(units=128, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(units=96, kernel_initializer=init, activation='tanh'))
model.compile(loss='mse', optimizer=simple_adam, metrics=['accuracy'])
model.fit(train_x, train_y, epochs=1000, batch_size=7, shuffle=True, verbose=True)
# Epoch 1/1000
# 351/351 [==============================] - 1s 3ms/step - loss: 0.0676 - accuracy: 0.0427
# Epoch 2/1000
# 351/351 [==============================] - 1s 2ms/step - loss: 0.0324 - accuracy: 0.0969
# Epoch 3/1000
# 351/351 [==============================] - 1s 3ms/step - loss: 0.0244 - accuracy: 0.0969
# ......
# Epoch 1000/1000
# 351/351 [==============================] - 1s 2ms/step - loss: 0.0028 - accuracy: 0.3476

pred_y = model.predict(test_x)
pred_y = (pred_y * t0_ptp) + t0_min

print(pred_y)
# array([[17.21479217, 17.52549491, 16.23683023, ..., 25.44682934,
#         23.65845003, 22.57863959],
#        [17.59345908, 18.73412209, 18.32082916, ..., 23.52446484,
#         22.62423444, 24.00765746],
#        [19.46384419, 17.62775962, 17.64109587, ..., 21.56474039,
#         22.1913017 , 22.44140341],
#         ...,
#        [18.96772808, 18.04357132, 16.24276439, ..., 23.13698257,
#         21.4985556 , 20.17896615],
#        [16.42696796, 14.55372871, 15.53627254, ..., 24.87524011,
#         22.83212658, 21.91864435],
#        [20.92249679, 22.7960805 , 21.07372294, ..., 19.58472966,
#         18.82105861, 20.52733855]])

print(pred_y.shape)
# (30, 96)

base = 0
error = 0
predates = data.index[-30:data.shape[0]].values
plt.figure(figsize=(20, 10))
for k in range(30):
    pred_array = pred_y[k]
    real_array = test_y_real.iloc[k].values
    plt.subplot(5, 7, k + 1)
    plt.title(predates[k])
    plt.plot(range(96), real_array, '-', label="real", c='black')
    plt.plot(range(96), pred_array, '--', label="pred", c='gray')
    base = base + np.sum(real_array)
    error = error + np.sum(np.abs(real_array - pred_array))
    plt.ylim(0, 250)

plt.show()
v = 100 * (1 - error / base)
print("Evaluation on test data: accuracy = %0.2f%% \n" % v)
# Evaluation on test data: accuracy = 79.66%
