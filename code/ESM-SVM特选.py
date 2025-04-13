from sklearn.model_selection import train_test_split,StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import roc_curve,auc,confusion_matrix,accuracy_score,roc_auc_score,matthews_corrcoef
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib
#输出结果无省略号
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

df = pd.read_csv("ESM.csv")

data = df.iloc[:, 1:]
target = df.iloc[:, 0]
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=66)
#使用smote过采样
smote = SMOTE(random_state=42)
# 将训练集进行SMOTE处理
x_train, y_train= smote.fit_resample(x_train,y_train)

# LightGBM特征选择
lgb_clf = lgb.LGBMClassifier()
lgb_clf.fit(x_train, y_train)
feature_importances = lgb_clf.feature_importances_
selected_features = data.columns[np.argsort(feature_importances)[::-1][:1100]]
x_train = x_train[selected_features]
x_test = x_test[selected_features]

# x_test.to_csv("ESM_SVM_test.csv", index=False)
# y_test.to_csv("ESM_SVMy_test.csv", index=False)
# # 建立模型
# model = svm.SVC(probability=True)
# #网格搜索
# params={
# 'kernel':('linear','rbf','poly'),
# 'C':[0.01,0.1,1,10,100]
# }
# grid_search = GridSearchCV(estimator=model
#                            ,param_grid=params
#                            , cv=5
#                            ,n_jobs=1
#                            ,scoring='roc_auc'
#                            ,verbose=2)
# grid_search.fit(x_train, y_train)
# # 输出最佳参数
# print("模型最优参数: ", grid_search.best_params_)
# print('最优模型对象：',grid_search.best_estimator_)
# grid1=grid_search.best_estimator_


# 创建模型
model = svm.SVC(kernel='poly',C=100,probability=True)
model.fit(x_train,y_train)
保存模型

joblib.dump(model, 'ESM-SVM-S.joblib')
# 预测
y_pred = model.predict(x_test)

# 计算准确率
print(f"准确率为：{accuracy_score(y_pred,y_test)}")
y_score = model.predict_proba(x_test)[:, 1]
# 计算不同阈值下的TPR和FPR
fpr, tpr, _ = roc_curve(y_test, y_score,pos_label=1)
# 计算AUC
roc_auc = auc(fpr, tpr)
print(f"AUC = {roc_auc}")

def calculate_sn_sp_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    sn = TP / (TP + FN)
    sp = TN / (TN + FP)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return sn, sp, mcc
sn, sp, mcc = calculate_sn_sp_mcc(y_test, y_pred)
print("SN:", sn)
print("SP:", sp)
print("MCC:", mcc)

