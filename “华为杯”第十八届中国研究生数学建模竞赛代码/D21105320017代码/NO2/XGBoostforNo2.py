#coding=utf-8
import os
import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression  #逻辑回归
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from pylab import mpl
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score  #交叉验证模块
from matplotlib import rcParams
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
# def print_K(model, x_train_data, y_train_data):
#     kf = KFold(n_splits=10)
#     rf_scores = cross_val_score(model, x_train_data, y_train_data, cv=10, scoring='neg_mean_squared_error')
#     mse_scores = -rf_scores
#     rmse_scores = np.sqrt(mse_scores)
#     print(rmse_scores.mean())
#     print(np.average(rf_scores))
# 加载数据，子表1和子表2
data = pd.read_excel('../datasets/Molecular_Descriptor.xlsx', sheet_name=0)
test_data = pd.read_excel('../datasets/Molecular_Descriptor.xlsx', sheet_name=1)

# 筛选出的20个特征
columns =["MDEC-23", "LipoaffinityIndex", "maxsOH", "nT6Ring", "minsssN", "BCUTp-1h", "C2SP2", 
        "hmin", "AMR", "SwHBa", "MDEC-22", "SP-5", "CrippenLogP", "C1SP2", "ATSp4", "ETA_dEpsilon_C",
        "MLFER_A", "C3SP2","MLogP","nC", "pIC50"]
test_columns = ["MDEC-23", "LipoaffinityIndex", "maxsOH", "nT6Ring", "minsssN", "BCUTp-1h", "C2SP2", 
        "hmin", "AMR", "SwHBa", "MDEC-22", "SP-5", "CrippenLogP", "C1SP2", "ATSp4", "ETA_dEpsilon_C",
        "MLFER_A", "C3SP2","MLogP","nC"]
data = data[columns]
test_data = test_data[test_columns]
# data.to_excel("../datasets/data_M_20.xlsx")
# test_data.to_excel("test_data_20.xlsx", header=None, index=None)

x = data.loc[:, data.columns != 'pIC50']
y = data.loc[:, 'pIC50']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
# print(x_train)
# print(x_test)
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(2, 15, 1)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators, 
'max_depth': max_depth, 
'min_child_weight':range(1,9,1)
}

model = XGBRegressor(n_estimators= 100, max_depth=8)

model.fit(x_train, y_train)
#返回最优的训练器
# best_estimator = rf_random.best_estimator_
# print(best_estimator)
# #输出最优训练器的精度
# print(rf_random.best_score_)
score = model.score(x_test, y_test)

# XGBoost模型预测
y_pred = model.predict(x_test) 
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
mae = metrics.mean_absolute_error(y_test, y_pred)
print("MAE: "+str(mae))
print("MSE", mean_squared_error(y_test, y_pred))
print('R^2: ', r2_score(y_test, y_pred))

plt.figure()
plt.plot(np.arange(len(y_test)), y_test,'co-',label='真实值', linewidth=0.8)
plt.plot(np.arange(len(y_test)), y_pred,'ro-',label='预测值', linewidth=0.8)
plt.title('XGBoostRegression')
plt.xlabel('测试样本')
plt.ylabel('pIC50')
plt.legend()
# plt.show()
def scatter_plot(TureValues,PredictValues):
    #绘图
    plt.figure()
    # plt.plot(xxx , yyy , c='0' , linewidth=1 , linestyle=':' , marker='.' , alpha=0.3)#绘制虚线
    plt.scatter(TureValues , PredictValues , s=20 ,  marker='o')#绘制散点图，横轴是真实值，竖轴是预测值

    plt.text(4, 7, r'$R^2=0.7314$',fontdict={'size':'16','color':'black'})
    # plt.xlim((0,1))   # 设置坐标轴范围
    # plt.ylim((0,1))
    plt.xlabel('真实的pIC50值')
    plt.ylabel('预测的pIC50值')
    plt.title('XGBoost模型')
    plt.show()
scatter_plot(y_test, y_pred)  # 生成散点图

yyy = model.predict(test_data) 
res = np.power(10, 9-yyy)
test_data['IC50'] = res 
test_data.to_excel('results.xlsx')

