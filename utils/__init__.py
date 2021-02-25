import copy
import datetime
import keras
import missingno
import networkx as nx
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import random
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.stattools as stat
from apyori import apriori
from fim import eclat
from keras.layers import SimpleRNN, Dense, LSTM
from keras.models import Sequential
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyecharts import options as opts
from pyecharts.charts import Kline
from rpy2.robjects import FloatVector, StrVector, r, ListVector
from rpy2.robjects.packages import importr
from scipy import stats
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.optimize import linprog
from scipy.spatial.distance import correlation as d_cor, pdist
from scipy.stats import chi2_contingency
from sklearn import linear_model, tree
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
from utils.data_path import AIR_MILES, \
    IRIS, MT_CARS, LAKE_HURON, AIR_PASSENGERS, ZAKI, AGR_INDEX, BOYS, CANADA, \
    ENERGY_OUT, WEATHER, GAME_CHURN, IRIS, CEMHT, M_SET, WINE, SUN_SPOT, WINE_IND
from utils.udf import arules_parse, \
    gains, disc, gains_ratio, eval_func, get_split_value, pair_plot, corr_plot, ccf, \
    g, times, add, log, gen_full_tree_exp, gen_side_tree_exp, random_get_tree, transform, \
    plot_tree, gen_individuals, inter_cross, mutate, evaluation_regression, get_adjust

# Mac/Windows 系统，使用如下设置可解决图表中文乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
