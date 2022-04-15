import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# data = pd.read_csv('data1.csv')

data = pd.read_excel('../datasets/1.xlsx', header=None, index_col=False)


# name = ["MDEC-23", "MLogP", "LipoaffinityIndex", "maxsOH", "minsOH", "nC", "nT6Ring", "n6Ring", "minsssN", "BCUTp-1h", "C2SP2", "hmin", "AMR", "SwHBa",
#         "maxsssN", "MDEC-22", "SP-5", "SaaCH", "CrippenLogP", "maxHsOH", "C1SP2", "nHaaCH", "naaCH", "ATSp4", "minHsOH", "ATSp2", "ATSp1", "SP-6", "nHother", "VAdjMat"]

list = ["MDEC-23", "LipoaffinityIndex", "maxsOH", "nT6Ring", "minsssN", "BCUTp-1h", "C2SP2",
        "hmin", "AMR", "SwHBa", "MDEC-22", "SP-5", "CrippenLogP", "C1SP2", "ATSp4", "ETA_dEpsilon_C",
        "MLFER_A", "C3SP2", "MLogP", "nC"]

data.columns = list

for it in list:
    # sns.pairplot(data[[it]], size=2.5)
    data[[it]].hist(bins=100)
    # plt.show()
    plt.ylabel('frequency')
    plt.show()
