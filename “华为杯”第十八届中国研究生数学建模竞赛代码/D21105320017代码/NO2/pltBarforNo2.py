import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号#有中文出现的情况，需要u'内容'
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
plt.figure(figsize=(12, 4))
# 构造x轴刻度标签、数据
labels = ['XGBoost', 'SVR', 'RandomForest', 'GBR']
first = [0.7314, 0.2293, 0.7247, 0.6731]# R2
# second = [0.5512, 1.5815, 0.565, 0.6709] #mse
third = [0.7424, 1.2576, 0.7516, 0.8191] # rmse
fourth = [0.5438, 1.033, 0.5517, 0.6135] # mae
# fifth = [26, 31, 35, 27, 21]




# 三组数据
plt.subplot(131)
x = np.arange(len(labels)) # x轴刻度标签位置
# width = 0.25 # 柱子的宽度
# 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# x - width，x， x + width即每组数据在x轴上的位置
plt.bar(x, first)
# plt.bar(x, second, width, label='2')
# plt.bar(x + width, third, width, label='3')
plt.ylabel(r'R^2')
plt.title('R-Square')
# x轴刻度标签位置不进行计算
plt.xticks(x, labels=labels)
# plt.legend()



# 三组数据
# plt.subplot(222)
# x = np.arange(len(labels)) # x轴刻度标签位置
# # width = 0.25 # 柱子的宽度
# # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# # x - width，x， x + width即每组数据在x轴上的位置
# plt.bar(x, second)
# # plt.bar(x, second, width, label='2')
# # plt.bar(x + width, third, width, label='3')
# plt.ylabel('MSE')
# plt.title('MSE')
# # x轴刻度标签位置不进行计算
# plt.xticks(x, labels=labels)
# # plt.legend()


# 三组数据
plt.subplot(132)
x = np.arange(len(labels)) # x轴刻度标签位置
# width = 0.25 # 柱子的宽度
# 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# x - width，x， x + width即每组数据在x轴上的位置
plt.bar(x, third)
# plt.bar(x, second, width, label='2')
# plt.bar(x + width, third, width, label='3')
plt.ylabel('RMSE')
plt.title('RMSE')
# x轴刻度标签位置不进行计算
plt.xticks(x, labels=labels)
# plt.legend()


# 三组数据
plt.subplot(133)
x = np.arange(len(labels)) # x轴刻度标签位置
# width = 0.25 # 柱子的宽度
# 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# x - width，x， x + width即每组数据在x轴上的位置
plt.bar(x, fourth)
# plt.bar(x, second, width, label='2')
# plt.bar(x + width, third, width, label='3')
plt.ylabel('MAE')
plt.title('MAE')
# x轴刻度标签位置不进行计算
plt.xticks(x, labels=labels)
# plt.legend()

plt.show()
