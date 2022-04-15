# coding=utf-8
from sklearn import preprocessing
from sko.PSO import PSO
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
import os
import sys
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from pylab import mpl
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from pandas.plotting import radviz

x = pd.read_excel(r"../datasets/1.xlsx", header=None, index_col=False)
t = preprocessing.StandardScaler().fit(x)
x = t.transform(x)
x = pd.DataFrame(x)

km = KMeans(n_clusters=2, n_init=10, n_jobs=-1)
km.fit(x)
print(km.score(x))

list = ["MDEC-23", "LipoaffinityIndex", "maxsOH", "nT6Ring", "minsssN", "BCUTp-1h", "C2SP2",
        "hmin", "AMR", "SwHBa", "MDEC-22", "SP-5", "CrippenLogP", "C1SP2", "ATSp4", "ETA_dEpsilon_C",
        "MLFER_A", "C3SP2", "MLogP", "nC"]
x.columns = list
x = pd.merge(x, pd.DataFrame(km.labels_, columns=[
             'class']), left_index=True, right_index=True)

radviz(x, class_column="class")
plt.show()

pd.plotting.andrews_curves(x, class_column="class")
plt.show()

# pd.plotting.parallel_coordinates(x, 'class')
# plt.show()
