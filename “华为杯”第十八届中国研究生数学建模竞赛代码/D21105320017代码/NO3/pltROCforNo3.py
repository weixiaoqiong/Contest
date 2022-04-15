
import numpy as np
import pandas as pd
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
# import matplotlib.pyplot as plt
# from xgboost.sklearn import XGBClassifier
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# , GradientBoostingClassifier
# import lightgbm as lgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
# from sklearn.preprocessing import StandardScaler

# lr_model = LogisticRegressionCV(class_weight='balanced', cv=5, max_iter=1000)
# svm_model = SVC(class_weight='balanced', gamma='auto', probability=True)
svm_model = SVC(kernel = 'rbf', gamma='scale', class_weight='balanced', probability=True)
# dt_model = DecisionTreeClassifier(class_weight='balanced')
# rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100)
# gbdt_model = GradientBoostingClassifier(n_estimators=100)
xg_model = XGBClassifier(n_estimators=250, max_depth=15)
# lgb_model = lgb.LGBMClassifier(n_estimators=100)

models = {
    # 'LR': lr_model,
    'SVM': svm_model,
    # 'DT': dt_model,
    # 'RF': rf_model,
    # 'GBDT': gbdt_model,
    'XGBoost': xg_model,
    # 'LightGBM': lgb_model
    }
df = pd.read_excel('../datasets/Molecular_Descriptor_ADMET.xlsx')
X = df.iloc[:, 1:-5]
# X = X.loc[:, (X != 0).any(axis=0)]
print(X)
y = df.loc[:, 'Caco-2']
# y = df.loc[:, 'hERG']
# y = df.loc[:, 'CYP3A4']
# y = df.loc[:, 'HOB']
# y = df.loc[:, 'MN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
def get_metric(clf, X, y_true):
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

    acc = metrics.accuracy_score(y_true, y_pred)
    p = metrics.precision_score(y_true, y_pred)
    r = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba[:, 1])
    auc = metrics.auc(fpr, tpr)
    return acc, p, r, f1, fpr, tpr, auc
df_result = pd.DataFrame(columns=('Model', 'dataset', 'Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC'))
row = 0
fprs_train = []
tprs_train = []
aucs_train = []
fprs_test = []
tprs_test = []
aucs_test = []
for name, clf in models.items():
    clf.fit(X_train, y_train)
    acc, p, r, f1, fpr_train, tpr_train, auc_train = get_metric(clf, X_train, y_train)
    fprs_train.append(fpr_train)
    tprs_train.append(tpr_train)
    aucs_train.append(auc_train)
    df_result.loc[row] = [name, 'train', acc, p, r, f1, auc_train]
    row += 1

    acc, p, r, f1, fpr_test, tpr_test, auc_test = get_metric(clf, X_test, y_test)
    fprs_test.append(fpr_test)
    tprs_test.append(tpr_test)
    aucs_test.append(auc_test)
    df_result.loc[row] = [name, 'test', acc, p, r, f1, auc_test]
    row += 1

    plt.figure()
    lw = 2
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=lw, label='train (AUC:%0.2f)' % auc_train)
    plt.plot(fpr_test, tpr_test, color='cornflowerblue', lw=lw, label='test (AUC:%0.2f)' % auc_test)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of '+name)
    plt.legend(loc="lower right")
    plt.savefig(name + '.png')
    plt.show()
df_result.to_csv('../datasets/df_result_MN_729.csv')