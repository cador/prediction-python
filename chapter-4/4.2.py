import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

x_list, y_list, type_list = [], [], []
plt.figure(figsize=(5, 5))

for e in [(2, 'o', False), (3, '+', True)]:
    r, symbol, flag = e
    x = np.linspace(-r, r, 30)
    y1 = np.sqrt(r ** 2 - x ** 2)
    y2 = -y1
    x_list = x_list + x.tolist() + x.tolist()
    y_list = y_list + y1.tolist() + y2.tolist()
    type_list = type_list + [flag] * (2 * len(x))
    plt.plot(x, y1, symbol, x, y2, symbol, color='black')

plt.show()
classify_data = pd.DataFrame({'x': x_list, 'y': y_list, 'type': type_list})
print(classify_data.head())

tm = LogisticRegression(solver='lbfgs')
X, y = classify_data.drop(columns=['type']), classify_data.type
tm.fit(X, y)
y_pred = tm.predict(X)
print(confusion_matrix(y, y_pred))
# 打印结果
# array([[60,  0],
#        [60,  0]], dtype=int64)

# 设置阶数为2和3阶
for i in [2, 3]:
    classify_data['x' + str(i)] = classify_data.x ** i
    classify_data['y' + str(i)] = classify_data.y ** i

X, y = classify_data.drop(columns=['type']), classify_data.type
tm.fit(X, y)
y_pred = tm.predict(X)
print(confusion_matrix(y, y_pred))
# 打印结果
# array([[60,  0],
#        [0 , 60]], dtype=int64)

coef = np.round(tm.coef_, 2)[0]
print(coef)
# array([-0.  ,  0.  ,  1.73,  1.8 , -0.  , -0.  ])

print(X.columns[np.where(coef > 0)])
# Index(['x2', 'y2'], dtype='object'))

plt.figure(figsize=(5, 5))
plt.plot(classify_data.x2, classify_data.y2, 'o')
plt.show()
