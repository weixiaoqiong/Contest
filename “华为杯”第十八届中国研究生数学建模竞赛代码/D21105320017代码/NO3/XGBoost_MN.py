import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
# from sklearn.datasets.samples_generator import make_blobs
# X, y = make_blobs()
df = pd.read_excel('../datasets/Molecular_Descriptor_ADMET.xlsx')
df_test = pd.read_excel('../datasets/Molecular_Descriptor.xlsx', sheet_name=1)

X = df.iloc[:, 1:-5]
# X = X.loc[:, (X != 0).any(axis=0)]
print(X)
# y = df.loc[:, 'Caco-2']
# y = df.loc[:, 'hERG']
# y = df.loc[:, 'CYP3A4']
# y = df.loc[:, 'HOB']
y = df.loc[:, 'MN']
# 140 0.90725
# 145 0.90387
# 150 0.90387
# pca = PCA(n_components=150, whiten=True, random_state=42) # 140
# model = SVC(kernel = 'rbf', gamma = 0.001)
# model = SVC(kernel = 'rbf', gamma='scale', class_weight='balanced', probability=True)
# model = make_pipeline(pca, svc)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# from sklearn.model_selection import GridSearchCV
# param_grid = {'svc__C':[1, 5, 10],
# 'svc__gamma':[0.0001, 0.0005, 0.001]}
# grid = GridSearchCV(model, param_grid)
from sklearn.ensemble import RandomForestClassifier
# pca = PCA(n_components=150, whiten=True, random_state=42) # 140

# model= RandomForestClassifier(n_estimators = 200, max_depth=8)
# model = make_pipeline(pca, RFC)
# 250 11
# 250 15
model= XGBClassifier(n_estimators = 250, max_depth=15)
# model= XGBClassifier(n_estimators = 250, max_depth=15)
# model= XGBClassifier(n_estimators = 250, max_depth=15)
# model= XGBClassifier(n_estimators = 100, max_depth=8)
# model.fit(x_train, y_train)
model.fit(x_train, y_train)
# print(grid.best_params_)
# model=grid.best_estimator_
y_pred = model.predict(x_test)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
# from numpy import mean
# print('Mean ROC AUC: %.3f' % np.mean(scores))
print(y_test)
print("预测")
print(y_pred)
score = model.score(x_test, y_test)
print(score)
# yfit.shape
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
# df_test = df_test.loc[:, (df_test != 0).any(axis=0)]
df_test = df_test.iloc[:, 1:]
print(df_test.head(5))
y_pred = model.predict(df_test)

# df_test['Caco-2'] = y_pred

# df = pd.read_excel('df_test.xlsx')

df = pd.DataFrame(y_pred)
name = 'MN'
# df['hERG'] = y_pred
df.to_excel('../datasets/df_test_result_ADMET_%s.xlsx'%name)

# y = df.loc[:, 'Caco-2']
# y = df.loc[:, 'hERG']
# y = df.loc[:, 'CYP3A4']
# y = df.loc[:, 'HOB']
# y = df.loc[:, 'MN']



# model.fit(X, y)
# def scatter_plot(TureValues,PredictValues):
#     #设置参考的1：1虚线参数
#     # xxx = [-0.5,1.5]
#     # yyy = [-0.5,1.5]
#     #绘图
#     plt.figure()
#     # print(len(TureValues))
#     # xxx = range(0, len(TureValues), 1)
#     # model0 = make_interp_spline(xxx, TureValues)
#     # yyy = model0(xxx)
#     # plt.plot(xxx , yyy , c='0' , linewidth=1 , linestyle=':' , marker='.' , alpha=0.3)#绘制虚线
#     plt.scatter(TureValues , PredictValues , s=20 ,  marker='o')#绘制散点图，横轴是真实值，竖轴是预测值

#     plt.text(4,9,r'$R^2=0.7314$',fontdict={'size':'16','color':'black'})
#     # plt.xlim((0,1))   #设置坐标轴范围
#     # plt.ylim((0,1))
#     # plt.xlabel('真实的pIC50值')
#     # plt.ylabel('预测的pIC50值')
#     plt.title('XGBoost模型')
#     plt.show()
# scatter_plot(y_test, y_pred)  #生成散点图





# from sklearn.ensemble import GradientBoostingClassifier
# gbm0= GradientBoostingClassifier(random_state=200)
# # GPCA = make_pipeline(pca, gbm0)
# GPCA.fit(x_train, y_train)
# yfit = GPCA.predict(x_test)
# score = GPCA.score(x_test, y_test)
# print(score)
# print("RMSE:", np.sqrt(mean_squared_error(y_test, yfit)))
# #获取模型返回值
# n_Support_vector = model.n_support_#支持向量个数
# Support_vector_index = model.support_#支持向量索引
# W = model.coef_#方向向量W
# b =  model.intercept_#截距项b
#绘制分类超平面

# def plot_point(dataArr,labelArr,Support_vector_index,W,b):
# 	if ax is None:
#         ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     x = np.linspace(xlim[0], xlim[1], 30)
#     y = np.linspace(ylim[0], ylim[1], 30)
#     Y, X = np.meshgrid(y, x)
#     xy = np.vstack([X.ravel(), Y.ravel()]).T
#     P = model.decision_function(xy).reshape(X.shape)
#     ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# 	plt.scatter(x,y,s=5,marker = 'h')
# 	plt.show()
# plot_point(X,labelArr,Support_vector_index,W,b)
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()

# y_score = model.predict_proba(x_test)

# # from sklearn.metrics import accuracy_score
# # score = accuracy_score(y_test, y_pred)
# # print("准确率：", score)
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print("混淆矩阵：", cm)
# #可视化上述混淆矩阵
# # plot_confusion_matrix(cm,classes = [0,1])
# from sklearn.metrics import precision_score
# pc = precision_score(y_test, y_pred) # 精准率
# print("Precision: ", pc)
# from sklearn.metrics import recall_score
# rs = recall_score(y_test, y_pred)
# print("Recall: ", rs)
# from sklearn.metrics import f1_score
# f1 = f1_score(y_test, y_pred) # F1 Score
# print("f1: ", f1)
# from sklearn.metrics import roc_curve
# print(y_test.values)
# # fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)        ### pos_label 表示 哪个 label 属于 正例
# print(y_score)
# fprs, tprs, thresholds = roc_curve(y_test.values, y_score[:,1])
# from sklearn.metrics import roc_curve, auc, roc_auc_score
# from sklearn.metrics import roc_auc_score

# y_true = np.array([0,1,1,0])
# y_score = np.array([0.85,0.78,0.69,0.54])
######## 计算 AUC ########
# roc_auc = auc(fprs, tprs)
# print(auc(fprs, tprs))            # 0.75      ### ROC 曲线下面积 AUC
# # 为每个类别计算ROC曲线和AUC
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# n_classes = 2
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])


# plt.figure()
# lw = 2
# color = ['r', 'g', 'b']
# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], color=color[i], lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.show()




