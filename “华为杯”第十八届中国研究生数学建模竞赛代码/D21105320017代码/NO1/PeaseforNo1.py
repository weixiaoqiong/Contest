import pandas as pd
import numpy as np
import math
from scipy.stats import stats
from pandas import DataFrame
# 问题一: 皮尔逊相关分析
# 加载数据
print('Load data...')
# data1.csv是处理后的数据(剔除全为0的列)
df = pd.read_csv('../datasets/data.csv')
n = df.shape[0]
m = df.shape[1] - 1
df_r = []
df_p = []
df_columns = list(df.columns)
del df_columns[0]
del df_columns[-1]
for i in range(1, m):
    r,p_value = stats.pearsonr(df.iloc[:, i], df['pIC50']) # 计算CRIM和target之间的相关系数和对应的显著性
    df_r.append(math.fabs(r))
    df_p.append(p_value)
c={"name" : df_columns,
    "r" :df_r, 
    "p" : df_p
    } # 将列表a，b转换成字典
data=DataFrame(c) # 将字典转换成为数据框
data.to_csv("../datasets/pease-绝对值.csv", index=False)