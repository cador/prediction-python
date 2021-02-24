from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, StrVector, r
from utils.data_path import GAME_CHURN
import pandas as pd

"""
:: 安装R包 => Python代码

import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=20)
utils.install_packages('CausalImpact')
utils.install_packages('zoo')

"""
ci = importr("CausalImpact")
zoo = importr("zoo")
base = importr("base")
gc_data = pd.read_csv(GAME_CHURN)
gc_data.y = [x.split('%')[0] for x in gc_data.y]
gc_data.orent = [x.split('%')[0] for x in gc_data.orent]
time_points = base.seq_Date(base.as_Date('2019-04-01'), by=1, length_out=25)
data = zoo.zoo(base.cbind(FloatVector(gc_data.y),
                          FloatVector(gc_data.cpi),
                          FloatVector(gc_data.orent)), time_points)
pre_period = base.as_Date(StrVector(['2019-04-01', '2019-04-15']))
post_period = base.as_Date(StrVector(['2019-04-16', '2019-04-25']))
model_args = r("list(niter = 10000, nseasons = 7, season.duration = 1)")
impact = ci.CausalImpact(data, pre_period, post_period, model_args=model_args)
r('pdf("../tmp/causal_impact.pdf")')
print(ci.plot_CausalImpact(impact))
r('dev.off()')
ci.PrintReport(impact)
