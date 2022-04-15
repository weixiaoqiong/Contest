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
from xgboost import XGBRegressor
from pylab import mpl
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pandas.plotting import radviz
# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


data = pd.read_excel('../datasets/data_M_20.xlsx')
# test_data = pd.read_excel('../datasets/Molecular_Descriptor.xlsx', sheet_name=1)

data_columns = ["MDEC-23", "LipoaffinityIndex", "maxsOH", "nT6Ring", "minsssN", "BCUTp-1h", "C2SP2",
        "hmin", "AMR", "SwHBa", "MDEC-22", "SP-5", "CrippenLogP", "C1SP2", "ATSp4", "ETA_dEpsilon_C",
        "MLFER_A", "C3SP2","MLogP","nC"]
# test_columns = ['MDEC-23', 'MLogP', 'LipoaffinityIndex', 'maxsOH', 'nC', 'nT6Ring',
#         'minsssN', 'BCUTp-1h', 'C2SP2', 'hmin', 'AMR', 'SwHBa', 'MDEC-22', 'SP-5',
#         'SaaCH', 'CrippenLogP', 'maxHsOH']

# data = data[columns]
# test_data = test_data[test_columns]
print(data.head())

x = data.iloc[:, 0:-6]
XG_y = data.loc[:, 'pIC50']
SVM_y_1 = data.loc[:, 'Caco-2']  # Caco-2
SVM_y_2 = data.loc[:, 'CYP3A4']  # CYP3A4
SVM_y_3 = data.loc[:, 'hERG']  # hERG
SVM_y_4 = data.loc[:, 'HOB']  # HOB
SVM_y_5 = data.loc[:, 'MN']  # MN

print(x)
x_train, x_test, y_train, y_test = train_test_split(
    x, XG_y, test_size=0.3, random_state=42)
model = XGBRegressor(n_estimators=100, max_depth=8, random_state=42)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("XG_score: ", score)

from xgboost.sklearn import XGBClassifier
x_train, x_test, y_train, y_test = train_test_split(x, SVM_y_1, test_size = 0.3, random_state = 42)
# svc_1 = SVC(kernel = 'rbf', gamma = 0.0001, C = 10)
svc_1 = XGBClassifier(n_estimators=100, max_depth = 11, random_state = 42)
svc_1.fit(x_train, y_train)
score = svc_1.score(x_test, y_test)
print("svc_1: ", score)

x_train, x_test, y_train, y_test = train_test_split(x, SVM_y_2, test_size = 0.3, random_state = 42)
# svc_2 = SVC(kernel = 'rbf', gamma = 0.0001)
svc_2 = XGBClassifier(n_estimators=100, max_depth = 8, random_state = 42)
svc_2.fit(x_train, y_train)
score = svc_2.score(x_test, y_test)
print("svc_2: ", score)

x_train, x_test, y_train, y_test = train_test_split(x, SVM_y_3, test_size = 0.3, random_state = 42)
# svc_3 = SVC(kernel = 'rbf', gamma = 0.0001, C = 10)
svc_3 = XGBClassifier(n_estimators=100, max_depth = 8, random_state = 42)
svc_3.fit(x_train, y_train)
score = svc_3.score(x_test, y_test)
print("svc_3: ", score)

x_train, x_test, y_train, y_test = train_test_split(x, SVM_y_4, test_size = 0.3, random_state = 42)
# svc_4 = SVC(kernel = 'rbf', gamma = 0.0001, C = 10)
svc_4 = XGBClassifier(n_estimators=100, max_depth = 8)
svc_4.fit(x_train, y_train)
score = svc_4.score(x_test, y_test)
print("svc_4: ", score)

x_train, x_test, y_train, y_test = train_test_split(x, SVM_y_5, test_size = 0.3, random_state = 42)
# svc_5 = SVC(kernel = 'rbf', gamma = 0.001, C=10)
svc_5 = XGBClassifier(n_estimators=100, max_depth = 8, random_state = 42)
svc_5.fit(x_train, y_train)
score = svc_5.score(x_test, y_test)
print("svc_5: ", score)

print(x.describe())


def lr(coordinate):
    x_data = DataFrame(coordinate).T
    print(x_data)
    x_data.columns = data_columns
    pred_Caco_2 = svc_1.predict(x_data)
    pred_CYP3A4 = svc_2.predict(x_data)
    pred_hERG = svc_3.predict(x_data)
    pred_HOB = svc_4.predict(x_data)
    pred_MN = svc_5.predict(x_data)
    count = 0
    if pred_Caco_2 == 1:
        count = count + 1
    if pred_CYP3A4 == 1:
        count = count + 1
    if pred_hERG == 0:
        count = count + 1
    if pred_HOB == 1:
        count = count + 1
    if pred_MN == 0:
        count = count + 1

    if count < 3:
        y_predict = 0
        # y_predict = model.predict(x_data)
        return(y_predict)
        # print(y_predict)
    else:
        y_predict = model.predict(x_data)
        # print("预测值:", y_predict)
        return(-y_predict)


print("Start PSO")
# 63.48345465,26.44632091,16.55797899,8.138222006,3.837642927,18.07531641,
# 28.39229452,0.587857508,548.9888586,58.70272731,44.41933407,29.33205426,
# 13.60378558,21.30146498,18390.12026,0.02778885,8.267461701,13.63937617,
# 6.427981205,101.6296789
# -9.463303585,-8.031519413,-4.087128938,-1.138222006,-1.102870809,
# 6.642373355,-4.392294519,-0.788274766,22.50754138,-30.49763146,
# -9.098628731,-0.254843689,-14.80005558,-1.301464976,-598.0927974,
# -0.39370885,-1.369461701,-1.639376166,0.782018795,0.37032107



# x_lb = [-9.463303585,-8.031519413,-4.087128938,-1.138222006,-1.102870809,
# 6.642373355,-4.392294519,-0.788274766,22.50754138,-30.49763146,
# -9.098628731,-0.254843689,-14.80005558,-1.301464976,-598.0927974,
# -0.39370885,-1.369461701,-1.639376166,0.782018795,0.37032107]
# x_ub = [63.48345465,26.44632091,16.55797899,8.138222006,3.837642927,18.07531641,
# 28.39229452,0.587857508,548.9888586,58.70272731,44.41933407,29.33205426,
# 13.60378558,21.30146498,18390.12026,0.02778885,8.267461701,13.63937617,
# 6.427981205,101.6296789]

x_lb = [-20.24389131,-7.470427218,-8.911416049,-2.520662,-2.253520125,
        -2.853050927,-9.595957659,-0.43228509,-68.13248618,-18.88956599,
        -19.45362605,-4.573108404,-3.444535881,-3.036784175,-2894.923632,
        -0.082893325,-1.132430344,-3.540792532,-1.457353752,-14.3215408]
x_ub = [36.5359302,13.18541,15.61135758,4.308670036,4.363704729,
        5.109102797,16.75780945,0.764675012,121.2242656,33.52630087,
        35.13814634,8.063997615,5.757097601,4.772005682,5067.454283,
        0.142459777,1.948339863,6.295464463,2.610533477,25.45653278]

pso = PSO(func=lr, dim=20, pop=1000, max_iter=100,
          # lb=x_lb, ub=x_ub, c1=0.3, c2=0.7, w=0.7)
            lb=x_lb, ub=x_ub, c1=1, c2=1, w=0.6)
pso.run()
print('Best_x is ', pso.gbest_x, 'Best_y is', pso.gbest_y)
# print('Best_x is ', pso.pbest_x, 'Best_y is', pso.pbest_y)
np.save("pbest_x.npy", pso.pbest_x)
np.save("pbest_y.npy", pso.pbest_y)

xxx = pd.DataFrame(pso.pbest_x)
writer = pd.ExcelWriter(r'1.xlsx')
xxx.to_excel(writer, 'sheet_1', float_format='%.2f', header=False, index=False)
writer.save()
writer.close()


xxx = pd.DataFrame(pso.pbest_y)
writer = pd.ExcelWriter(r'2.xlsx')
xxx.to_excel(writer, 'sheet_1', float_format='%.2f', header=False, index=False)
writer.save()
writer.close()



y_h = [-x for x in pso.gbest_y_hist]
plt.plot(y_h)
plt.show()


x = pso.pbest_x



t= preprocessing.StandardScaler().fit(x)#标准化
x=t.transform(x)
x = pd.DataFrame(x)

km = KMeans(n_clusters=2, n_init=10, n_jobs=-1)
km.fit(x)
print(km.score(x))
print(km.labels_)

xxx = pd.DataFrame(km.labels_)
writer = pd.ExcelWriter(r'3.xlsx')
xxx.to_excel(writer, 'sheet_1', float_format='%.2f', header=False, index=False)
writer.save()
writer.close()

print(km.cluster_centers_)



list = ["MDEC-23", "LipoaffinityIndex", "maxsOH", "nT6Ring", "minsssN", "BCUTp-1h", "C2SP2",
        "hmin", "AMR", "SwHBa", "MDEC-22", "SP-5", "CrippenLogP", "C1SP2", "ATSp4", "ETA_dEpsilon_C",
        "MLFER_A", "C3SP2","MLogP","nC"]
x = DataFrame(pso.pbest_x, columns=list)
x = pd.merge(x, pd.DataFrame(km.labels_, columns=[
             'class']), left_index=True, right_index=True)

radviz(x, class_column="class")
plt.show()