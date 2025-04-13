from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import shap
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, confusion_matrix, matthews_corrcoef
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
#读取数据
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
df = pd.read_csv("AAC.csv")
data = df.iloc[:, 1:]
target = df.iloc[:, 0]

#划分数据集
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=66)
# x_test.to_csv("AACx_test.csv", index=False)
# y_test.to_csv("AACy_test.csv", index=False)
#使用smote过采样
smote = SMOTE(random_state=42)
# 将训练集进行SMOTE处理
x_train, y_train= smote.fit_resample(x_train,y_train)
#
# # 定义模型
# model = CatBoostClassifier()
# # 定义超参数网格
# param_grid = {
#     'learning_rate':np.arange(0.07,0.17,0.01),
#     'depth':np.arange(7,15,1),
#     'iterations':[200]
# }
# # 定义网格搜索对象
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
# # 进行网格搜索
# grid_search.fit(x_train, y_train)
# # 获取最佳参数
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)
#Best Parameters: {'depth': 8, 'iterations': 200, 'learning_rate': 0.11999999999999998}
# # 创建模型
model = CatBoostClassifier(iterations=100, learning_rate=0.11, depth=11)
model.fit(x_train, y_train)
#保存模型
# joblib.dump(model, 'AAC-CB.joblib')
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