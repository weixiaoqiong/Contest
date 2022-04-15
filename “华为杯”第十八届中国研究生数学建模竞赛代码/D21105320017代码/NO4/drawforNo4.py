# coding=utf-8
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
from pylab import mpl
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from pandas.plotting import radviz
# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn import preprocessing
plt.style.use('ggplot')


x = pd.read_excel(r"../datasets/1.xlsx", header=None, index_col=False)


t = preprocessing.StandardScaler().fit(x)
x = t.transform(x)
x = pd.DataFrame(x)

km = KMeans(n_clusters=2, n_init=10, n_jobs=-1)
km.fit(x)

d=[]
for i in range(1,10):    #k取值1~11，做kmeans聚类，看不同k值对应的簇内误差平方和
    cluster = KMeans(n_clusters=i, n_init=10, n_jobs=-1)
    cluster.fit(x)
    print(cluster.score(x))
    d.append(cluster.inertia_)  #inertia簇内误差平方和

plt.plot(range(1,10),d,marker='o')
plt.xlabel('number of clusters')
plt.ylabel('distortions')
plt.show()

score = []
for i in range(2, 30):
    cluster = KMeans(n_clusters=i, n_init=10, n_jobs=-1)
    cluster.fit(x)
    score.append(silhouette_score(x, cluster.labels_))
plt.plot(range(2, 30), score)
plt.axvline(pd.DataFrame(score).idxmin()[0]+2, ls=':')  # x轴的取值起点是2，所以id+2
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')
plt.show()

score = []
for i in range(2, 6):
    cluster = KMeans(n_clusters=i, n_init=10, n_jobs=-1)
    cluster.fit(x)
    score.append(silhouette_score(x, cluster.labels_))
plt.plot(range(2, 6), score)
plt.axvline(pd.DataFrame(score).idxmin()[0]+2, ls=':')  # x轴的取值起点是2，所以id+2
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')
plt.show()

list = ["MDEC-23", "LipoaffinityIndex", "maxsOH", "nT6Ring", "minsssN", "BCUTp-1h", "C2SP2",
        "hmin", "AMR", "SwHBa", "MDEC-22", "SP-5", "CrippenLogP", "C1SP2", "ATSp4", "ETA_dEpsilon_C",
        "MLFER_A", "C3SP2", "MLogP", "nC"]
x.columns = list
x = pd.merge(x, pd.DataFrame(km.labels_, columns=[
             'class']), left_index=True, right_index=True)

radviz(x, class_column="class")
plt.show()
