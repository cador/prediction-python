from utils import *

# 准备基础数据
iris = pd.read_csv(IRIS)
x, y = iris.drop(columns=['Species', 'Petal.Width']), iris['Petal.Width']

# 标准化处理
x = x.apply(lambda v: (v - np.mean(v)) / np.std(v))
x = np.c_[[1] * x.shape[0], x]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

# 初始化精度控制参数 ε
epsilon = 4.0

# 初始化学习效率 α
alpha = 0.005

# 精度控制变量 d
d = epsilon + 1

# 用适当小的随机数初始化权向量 W
w = np.random.uniform(0, 1, 4)

while d >= epsilon:
    d = 0
    for i in range(x_train.shape[0]):
        xi = x_train[i, :]
        delta = np.sum(w * xi) - y_train.values[i]
        w = w - alpha * delta * xi
        d = d + delta ** 2

    print(d)

# 66.86549781394255
# 26.873553725350185
# ......
# 4.0241010398100885
# 3.9844824194932182

print(w)
# array([ 1.19918387, -0.05429913,  0.06292586,  0.80169231])

print(np.sum((y_test - np.sum(x_test * w, axis=1)) ** 2))
# 1.8217475510303391

# 准备基础数据
iris = pd.read_csv(IRIS)
x, y = iris.drop(columns=['Species', 'Petal.Width']), iris['Petal.Width']

# 标准化处理
x = x.apply(lambda v: (v - np.mean(v)) / np.std(v))
x = np.c_[[1] * x.shape[0], x]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

# 设定学习效率 alpha
alpha = 0.01

# 评估隐含层神经元个数
m = int(np.round(np.sqrt(0.43 * 1 * 4 + 0.12 + 2.54 * 4 + 0.77 + 0.35) + 0.51, 0))

# 初始化输入向量的权重矩阵
w_input = np.random.uniform(-1, 1, (m, 4))

# 初始化隐含层到输出的权重向量
w_hide = np.random.uniform(-1, 1, m)
epsilon = 1e-3
errorList = []

# 进入迭代
error = None
for p in range(1000):
    error = 0
    for i in range(x_train.shape[0]):
        # 正向传播过程
        x_input = x_train[i, :]
        d = np.matmul(w_input, x_input)
        z = (np.exp(d) - np.exp(-d)) / (np.exp(d) + np.exp(-d))
        o = np.matmul(w_hide, z)
        e = o - y_train.values[i]
        error = error + e ** 2
        # 若 e>epsilon，则进入反向传播过程
        if np.abs(e) > epsilon:
            w_hide = w_hide - alpha * z * e
            a = (4 * np.exp(2 * d) / ((np.exp(2 * d) + 1) ** 2)) * w_hide * alpha * e
            w_input = w_input - [x * x_input for x in a]

    errorList.append(error)
    print("iter:", p, "error:", error)

    # 当连续两次残差平方和的差小于epsilon时，退出循环
    if len(errorList) > 2 and errorList[-2] - errorList[-1] < epsilon:
        break

# iter: 0 error: 155.54395018394294
# iter: 1 error: 56.16645418105049
# iter: 2 error: 28.788174184286994
# ......
# iter: 141 error: 2.8748747912336405
# iter: 142 error: 2.873880717019784

print(error)
# 2.873880717019784

print(w_input)
# array([[ 1.42771077, -0.20451346, -0.11610576,  0.54878999],
#        [ 0.40080947,  0.67893308,  0.15667116, -0.56050505],
#        [ 0.15342243, -0.01495382,  0.19293603, -0.88798248],
#        [-0.52516692,  0.81847855, -0.46910003, -0.34153195]])

print(w_hide)
# array([ 0.91786267,  0.521612  , -1.32963221, -0.67729756])

y_pred = []
for i in range(x_test.shape[0]):
    # 正向传播过程
    x_input = x_test[i, :]
    d = np.matmul(w_input, x_input)
    z = (np.exp(d) - np.exp(-d)) / (np.exp(d) + np.exp(-d))
    o = np.matmul(w_hide, z)
    y_pred.append(o)

print(np.sum((y_test.values - y_pred) ** 2))
# 2.125832275861781

# 准备基础数据
iris = pd.read_csv(IRIS)
x, y = iris.drop(columns=['Species', 'Petal.Width']), iris['Petal.Width']

# 标准化处理
x = x.apply(lambda v: (v - np.mean(v)) / np.std(v))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

# 定义模型
init = keras.initializers.glorot_uniform(seed=1)
simple_adam = keras.optimizers.Adam(lr=0.0001)
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=8, input_dim=3, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dense(units=16, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dense(units=8, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dense(units=4, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dense(units=1, kernel_initializer=init, activation='relu'))
model.compile(loss='mean_squared_error', optimizer=simple_adam)

# 训练模型
b_size = 2
max_epochs = 100
print("Starting training ")
h = model.fit(x_train, y_train, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
print("Training finished \n")

# Starting training
# Epoch 1/100
# 100/100 [==============================] - 2s 20ms/step - loss: 2.0479
# ......
# Epoch 99/100
# 100/100 [==============================] - 0s 1ms/step - loss: 0.0429
# Epoch 100/100
# 100/100 [==============================] - 0s 1ms/step - loss: 0.0428
# Training finished

# 评估模型
out = model.evaluate(x_test, y_test, verbose=0)
print("Evaluation on test data: loss = %0.6f \n" % (out * len(y_test)))
# Evaluation on test data: loss = 1.869615
