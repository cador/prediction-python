import pandas_datareader.data as web
import datetime as dt
from pyecharts import options as opts
from pyecharts.charts import Kline

data = web.DataReader('600519.ss', 'yahoo', dt.datetime(2019, 8, 1), dt.datetime(2019, 8, 31))
data.head()
#               High         Low        Open       Close    Volume   Adj Close
# Date
# 2019-08-01    977.000000  953.020020  976.51001   959.299988  3508952 959.299988
# 2019-08-02    957.979980  943.000000  944.00000   954.450012  3971940 954.450012
# 2019-08-05    954.000000  940.000000  945.00000   942.429993  3677431 942.429993
# 2019-08-06    948.000000  923.799988  931.00000   946.299988  4399116 946.299988
# 2019-08-07    955.530029  945.000000  949.50000   945.000000  2686998 945.000000

kl_data = data.values[:, [2, 3, 1, 0]]  # 分别对应开盘价、收盘价、最低价和最高价

k_obj = Kline().add_xaxis(data.index.strftime("%Y-%m-%d").tolist()).add_yaxis("贵州茅台-日K线图",
                                                                              kl_data.tolist()).set_global_opts(
    yaxis_opts=opts.AxisOpts(is_scale=True), xaxis_opts=opts.AxisOpts(is_scale=True),
    title_opts=opts.TitleOpts(title=""))
k_obj.render("../tmp/贵州茅台股票日K线图.html")
