import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.metrics import r2_score
from sklearn import metrics
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
def print_K(model, x_train_data, y_train_data):
    kf = KFold(n_splits=10)
    rf_scores = cross_val_score(model, x_train_data, y_train_data, cv=10, scoring='neg_mean_squared_error')
    mse_scores = -rf_scores
    rmse_scores = np.sqrt(mse_scores)
    print(rmse_scores.mean())
    print(np.average(rf_scores))

data = pd.read_excel('../datasets/Molecular_Descriptor.xlsx', sheet_name=0)
test_data = pd.read_excel('../datasets/Molecular_Descriptor.xlsx', sheet_name=1)

# columns = ['MDEC-23', 'MLogP', 'LipoaffinityIndex', 'maxsOH', 'nC', 'nT6Ring', 'minsssN', 'BCUTp-1h', 'C2SP2', 
#         'hmin', 'AMR', 'SwHBa', 'MDEC-22', 'SP-5', 'SaaCH', 'CrippenLogP', 'maxHsOH', 'C1SP2', 'nHaaCH', 'ATSp4', 'pIC50']
# columns = ['MDEC-23', 'LipoaffinityIndex', 'BCUTp-1h', 'maxsOH', 'MLogP', 'hmin', 'SwHBa', 'ATSc3', 
#         'ATSc4', 'CrippenLogP', 'SaasC', 'XLogP', 'BCUTc-1l', 'MDEC-22', 'SP-5', 
#         'minsssN', 'SaaCH', 'AMR', 'minHBint10', 'maxHsOH', 'pIC50']
# columns = ['MDEC-23', 'MLogP', 'LipoaffinityIndex', 'maxsOH', 'minsOH', 'nC', 'nT6Ring', 
#         'n6Ring', 'minsssN', 'BCUTp-1h', 'C2SP2', 'hmin', 'AMR', 'SwHBa', 'maxsssN', 'MDEC-22', 'SP-5', 
#         'SaaCH', 'CrippenLogP', 'maxHsOH', 'pIC50']
# test_columns = ['MDEC-23', 'MLogP', 'LipoaffinityIndex', 'maxsOH', 'nC', 'nT6Ring', 'minsssN', 'BCUTp-1h', 'C2SP2', 
#                 'hmin', 'AMR', 'SwHBa', 'MDEC-22', 'SP-5', 'SaaCH', 'CrippenLogP', 'maxHsOH', 'C1SP2', 'nHaaCH', 'ATSp4']
# test_columns = ['MDEC-23', 'LipoaffinityIndex', 'BCUTp-1h', 'maxsOH', 'MLogP', 'hmin', 'SwHBa', 'ATSc3', 'ATSc4', 'CrippenLogP', 
#         'SaasC', 'XLogP', 'BCUTc-1l', 'MDEC-22', 'SP-5', 'minsssN', 'SaaCH', 'AMR', 'minHBint10', 'maxHsOH']

# columns = ['MDEC-23', 'MLogP', 'LipoaffinityIndex', 'maxsOH', 'nC', 'nT6Ring', 'minsssN', 
#         'BCUTp-1h', 'C2SP2', 'hmin', 'AMR', 'SwHBa', 'MDEC-22', 'SP-5', 'SaaCH', 'CrippenLogP', 
#         'maxHsOH', 'C1SP2', 'nHaaCH', 'ATSp4', 'pIC50']

# test_columns = ['MDEC-23', 'MLogP', 'LipoaffinityIndex', 'maxsOH', 'nC', 'nT6Ring', 'minsssN', 
#         'BCUTp-1h', 'C2SP2', 'hmin', 'AMR', 'SwHBa', 'MDEC-22', 'SP-5', 'SaaCH', 'CrippenLogP', 
#         'maxHsOH', 'C1SP2', 'nHaaCH', 'ATSp4']

columns =["MDEC-23", "LipoaffinityIndex", "maxsOH", "nT6Ring", "minsssN", "BCUTp-1h", "C2SP2", 
        "hmin", "AMR", "SwHBa", "MDEC-22", "SP-5", "CrippenLogP", "C1SP2", "ATSp4", "ETA_dEpsilon_C",
        "MLFER_A", "C3SP2","MLogP","nC", "pIC50"]
test_columns = ["MDEC-23", "LipoaffinityIndex", "maxsOH", "nT6Ring", "minsssN", "BCUTp-1h", "C2SP2", 
        "hmin", "AMR", "SwHBa", "MDEC-22", "SP-5", "CrippenLogP", "C1SP2", "ATSp4", "ETA_dEpsilon_C",
        "MLFER_A", "C3SP2","MLogP","nC"]
data = data[columns]


x = data.loc[:, data.columns != 'pIC50']
y = data.loc[:, 'pIC50']
print(x.head(5))
print(y.head(5))

###########1.数据生成部分##########
# def f(x1, x2):
#     y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2) + 3 + 0.1 * x1 
#     return y

# def load_data():
#     x1_train = np.linspace(0,50,500)
#     x2_train = np.linspace(-10,10,500)
#     data_train = np.array([[x1,x2,f(x1,x2) + (np.random.random(1)-0.5)] for x1,x2 in zip(x1_train, x2_train)])
#     x1_test = np.linspace(0,50,100)+ 0.5 * np.random.random(100)
#     x2_test = np.linspace(-10,10,100) + 0.02 * np.random.random(100)
#     data_test = np.array([[x1,x2,f(x1,x2)] for x1,x2 in zip(x1_test, x2_test)])
#     return data_train, data_test

# train, test = load_data()
# x_train, y_train = train[:,:2], train[:,2] #数据前两列是x1,x2 第三列是y,这里的y有随机噪声
# x_test ,y_test = test[:,:2], test[:,2] # 同上,不过这里的y没有噪声
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
l_score = []
l_mse = []
l_rmse = []
l_mae= []
var = 221
plt.figure(1, figsize=(10, 10))
###########2.回归部分##########
def try_different_method(model, nnnn):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    print("得分:", score)
    l_score.append(round(score, 4))
    print('R^2: ', r2_score(y_test, result))
    mse = metrics.mean_squared_error(y_test, result)
    print("mse: ", mse)
    l_mse.append(round(mse, 4))
    rmse = np.sqrt(metrics.mean_squared_error(y_test, result)) 
    l_rmse.append(round(rmse,4))
    print("rmse: "+str(rmse))
    mae = metrics.mean_absolute_error(y_test, result)
    l_mae.append(round(mae,4))
    print("mae: "+str(mae))
    print('\n')
    # mmape = mape(y_test, result) 
    # print("mape"+str(mmape))
    global var
    # 拟合图
    plt.subplot(var)
    # plt.plot(np.arange(len(result)), y_test,'co-',label='真实值')
    # plt.plot(np.arange(len(result)),result,'ro-',label='预测值')
    # plt.title(nnnn)
    # plt.xlabel('测试样本')
    # plt.ylabel('pIC50')
    # plt.legend()


    # plt.show()
    xxx = 4
    yyy = 9
    plt.scatter(y_test, result , s=20 ,  marker='o')#绘制散点图，横轴是真实值，竖轴是预测值
    if var == 222:
        yyy = 7
    plt.text(xxx, yyy, r'$R^2=%.4lf$'%score,fontdict={'size':'12','color':'black'})
    # plt.xlim((0,1))   #设置坐标轴范围
    # plt.ylim((0,1))
    if var == 223 or var == 224:
        plt.xlabel('真实的pIC50值')
    if var == 221 or var == 223:
        plt.ylabel('预测的pIC50值')
    
    plt.title(nnnn)
    var = var + 1
    # plt.show()


###########2.回归部分##########
def try_different_method2(model, nnnn):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    print("得分:", score)
    l_score.append(round(score, 4))
    print('R^2: ', r2_score(y_test, result))
    mse = metrics.mean_squared_error(y_test, result)
    print("mse: ", mse)
    l_mse.append(round(mse, 4))
    rmse = np.sqrt(metrics.mean_squared_error(y_test, result)) 
    l_rmse.append(round(rmse,4))
    print("rmse: "+str(rmse))
    mae = metrics.mean_absolute_error(y_test, result)
    l_mae.append(round(mae,4))
    print("mae: "+str(mae))
    print('\n')
    # mmape = mape(y_test, result) 
    # print("mape"+str(mmape))
    global var
    # 拟合图
    plt.subplot(var)
    plt.plot(np.arange(len(result)), y_test,'co-',label='真实值')
    plt.plot(np.arange(len(result)),result,'ro-',label='预测值')
    plt.title(nnnn)
    plt.xlabel('测试样本')
    plt.ylabel('pIC50')
    plt.legend()

    # plt.show()
    var = var + 1
    # plt.show()
###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR(kernel='rbf')
# from sklearn.svm import SVR
# clf = SVR(kernel='rbf', C=1.25)
# clf.fit(x_tran, y_train)
# y_hat = clf.predict(x_test)
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=100)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()


###########4.具体方法调用部分##########
model_XGBT = XGBRegressor(n_estimators= 100, max_depth=8)
try_different_method(model_XGBT, 'XGBoost')
try_different_method(model_SVR, 'SVR')
# try_different_method(model_LinearRegression)
# try_different_method(model_KNeighborsRegressor)
try_different_method(model_RandomForestRegressor, 'RandomForest')
# try_different_method(model_AdaBoostRegressor)
try_different_method(model_GradientBoostingRegressor, 'GBDT')
# try_different_method(model_BaggingRegressor)
# try_different_method(model_ExtraTreeRegressor)
print("得分: ", l_score)
print("mse: ", l_mse)
print("rmse:", l_rmse)
print("mae: ", l_mae)

plt.figure(2, figsize=(10, 10))
var=221
try_different_method2(model_XGBT, 'XGBoost')
try_different_method2(model_SVR, 'SVR')

try_different_method2(model_RandomForestRegressor, 'RandomForest')

try_different_method2(model_GradientBoostingRegressor, 'GBDT')
plt.show()

