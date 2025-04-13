from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

df = pd.read_csv("CTDC.csv")

data = df.iloc[:, 1:]
target = df.iloc[:, 0]
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=66)
#使用smote过采样
smote = SMOTE(random_state=42)
# 将训练集进行SMOTE处理
x_train, y_train= smote.fit_resample(x_train,y_train)
# LightGBM特征选择
lgb_clf = lgb.LGBMClassifier(random_state=42)
lgb_clf.fit(x_train, y_train)
feature_importances = lgb_clf.feature_importances_
selected_features = data.columns[np.argsort(feature_importances)[::-1][:11]]  # 选择前20个重要特征
x_train = x_train[selected_features]
x_test = x_test[selected_features]

# x_test.to_csv("CTDC_RFx_test.csv", index=False)
# y_test.to_csv("CTDC_RFy_test.csv", index=False)
# # 定义模型
# model = RandomForestClassifier()
# # 定义超参数网格
# param_grid = {
#     'n_estimators':np.arange(100,400,100),
#     'max_depth':np.arange(3,14,1),
#     'min_samples_leaf':np.arange(1,10,1),
#     'min_samples_split':np.arange(1,10,1)
# }
# # 定义网格搜索对象
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
# # 进行网格搜索
# grid_search.fit(x_train, y_train)
# # 获取最佳参数
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)


# 创建模型
model = RandomForestClassifier(n_estimators=300,max_depth=13,min_samples_leaf=1,min_samples_split=2)
model.fit(x_train, y_train)
#保存模型

# joblib.dump(model, 'CTDC-RF-S.joblib')
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