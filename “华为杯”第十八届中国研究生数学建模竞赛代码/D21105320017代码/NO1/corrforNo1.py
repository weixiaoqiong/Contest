import seaborn
import matplotlib.pyplot as mp
import pandas as pd

# data = pd.read_excel('./data_M_20.xlsx')
# # data = pd.read_excel('Molecular_Descriptor.xlsx')
# df = pd.DataFrame(data[data.columns[:20]])
data = pd.read_excel('../datasets/Molecular_Descriptor_pIC50_ADMET.xlsx')
# data = pd.read_excel('1.xlsx', header=None, index_col=False)

# , "ETA_Epsilon_4", "SaasC", "nHeavyAtom", "SHaaCH", "nHsOH", "nsOH", "maxssCH2", "VP-0", "naasC", "nH", "mindO", "SHother", "ETA_Eta_L", "maxdO", "SaaN", "XLogP", "naaN", "SP-0", "naAromAtom", "SssCH2", "maxaaN", "SsssN", "minHssNH"]])
df = pd.DataFrame(data[["maxsOH", "minsOH", "nT6Ring",
                  "n6Ring", "minsssN", "maxsssN", "nHaaCH", "naaCH"]])

#

df_corr = df.corr()

print(df_corr)
seaborn.heatmap(df_corr, center=0, annot=True, cmap='YlGnBu')
mp.show()

list = ["MDEC-23", "LipoaffinityIndex", "maxsOH", "nT6Ring", "minsssN", "BCUTp-1h", "C2SP2",
        "hmin", "AMR", "SwHBa", "MDEC-22", "SP-5", "CrippenLogP", "C1SP2", "ATSp4", "ETA_dEpsilon_C",
        "MLFER_A", "C3SP2", "MLogP", "nC"]

# , "ETA_Epsilon_4", "SaasC", "nHeavyAtom", "SHaaCH", "nHsOH", "nsOH", "maxssCH2", "VP-0", "naasC", "nH", "mindO", "SHother", "ETA_Eta_L", "maxdO", "SaaN", "XLogP", "naaN", "SP-0", "naAromAtom", "SssCH2", "maxaaN", "SsssN", "minHssNH"]])
df = pd.DataFrame(data[list])

#

df_corr = df.corr()

print(df_corr)
seaborn.heatmap(df_corr, center=0, annot=True, cmap='YlGnBu')
mp.show()
