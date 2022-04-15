import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号#有中文出现的情况，需要u'内容'
from matplotlib import rcParams
import pandas as pd
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

# df = pd.read_excel('../datasets/ADMET.xlsx')

# X = df.iloc[:, -5:]
# print(df.iloc[:,-5].value_counts())
# print(df.iloc[:,-4].value_counts())
# print(df.iloc[:,-3].value_counts())
# print(df.iloc[:,-2].value_counts())
# print(df.iloc[:,-1].value_counts())
# print(X)

plt.figure(figsize=(10, 10))
# 构造x轴刻度标签、数据
labels = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
first = [1215, 1461, 1099, 1465, 1514] # R2
second = [759, 513, 875, 509, 460] #mse
# third = [0.7636, 1.04270, 1.16068, 0.75530, 0.8374] # rmse
# fourth = [0.56402, 0.82928, 0.92537, 0.55615, 0.62677] # mae
# fifth = [26, 31, 35, 27, 21]




# 三组数据
plt.subplot(221)
x = np.arange(len(labels)) # x轴刻度标签位置
width = 0.3 # 柱子的宽度0.25
# 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# x - width，x， x + width即每组数据在x轴上的位置
plt.bar(x-width/2, first, width, label='0')
plt.bar(x+ width/2, second, width, label='1')
# plt.bar(x + width, third, width, label='3')
plt.ylabel('个数')
plt.title('数据特征')
# x轴刻度标签位置不进行计算
plt.xticks(x, labels=labels)
plt.legend()
plt.show()