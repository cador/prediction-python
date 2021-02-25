from utils import *

out = pd.read_csv(M_SET)
print(out.head())
#      y    x1   x2
# 0  16.3    1.1 1.1
# 1  16.8    1.4 1.5
# 2  19.2    1.7 1.8
# 3  18.0    1.7 1.7
# 4  19.5    1.8 1.9

r_squared_i = ols("x1~x2", data=out).fit().rsquared
vif = 1. / (1. - r_squared_i)
print(vif)
# 35.962864339690476
