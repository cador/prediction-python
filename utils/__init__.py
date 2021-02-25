from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot as plt
from utils.data_path import WINE_IND
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils.data_path import AIR_MILES, IRIS, MT_CARS, LAKE_HURON
from matplotlib import pyplot as plt
from utils.udf import pair_plot, corr_plot, ccf
import pandas as pd
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cross_decomposition import CCA
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, StrVector, r
from utils.data_path import GAME_CHURN
import pandas as pd
from sklearn.cluster import KMeans
from utils.data_path import AIR_PASSENGERS
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from apyori import apriori
import networkx as nx
from fim import eclat
import numpy as np
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
from rpy2.robjects import StrVector, ListVector
from utils.data_path import AIR_PASSENGERS, ZAKI
from utils.udf import arules_parse
import pandas as pd
import numpy as np
from utils.data_path import BOYS
from matplotlib import pyplot as plt
from utils.data_path import IRIS
from utils.udf import get_split_value
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy import stats
from scipy.spatial.distance import correlation as d_cor
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from utils.data_path import IRIS, WINE
from utils.udf import gains, disc, gains_ratio, eval_func
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import copy
from utils.data_path import IRIS, CEMHT
from utils.udf import g, times, add, log, gen_full_tree_exp, gen_side_tree_exp, \
    random_get_tree, transform, plot_tree, gen_individuals, inter_cross, mutate, evaluation_regression, get_adjust
import random
import numpy as np
import pandas as pd
from sklearn import linear_model
from utils.data_path import IRIS
import pandas as pd
import numpy as np
from utils.data_path import IRIS
import random
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from utils.data_path import M_SET
import matplotlib.pyplot as plt
from utils.data_path import M_SET, IRIS
from sklearn.linear_model import RidgeCV
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from utils.data_path import IRIS
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from utils.data_path import IRIS
import pandas as pd
import numpy as np
from sklearn import linear_model
import random
from utils.data_path import IRIS
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from utils.data_path import IRIS
from utils.data_path import IRIS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.data_path import IRIS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.data_path import IRIS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.data_path import AGR_INDEX
import statsmodels.tsa.stattools as stat
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_path import SUN_SPOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_path import CANADA
import statsmodels.tsa.stattools as stat
import pandas as pd
import numpy as np
from utils.data_path import CANADA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.data_path import CANADA
from keras.layers import SimpleRNN, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.data_path import CANADA
from keras.layers import LSTM, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import missingno
import pandas as pd
from matplotlib import pyplot as plt
from utils.data_path import ENERGY_OUT, WEATHER
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
import keras
import pandas_datareader.data as web
import datetime as dt
from pyecharts import options as opts
from pyecharts.charts import Kline
import statsmodels.tsa.stattools as stat
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from keras.layers import LSTM, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

# Mac/Windows 系统，使用如下设置可解决图表中文乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
