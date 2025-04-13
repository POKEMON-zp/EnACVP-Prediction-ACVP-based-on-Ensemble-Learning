from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
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

#读取数据
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
df = pd.read_csv("CTDC+EGAAC.csv")
data = df.iloc[:, 1:]
target = df.iloc[:, 0]

#划分数据集
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=66)
#使用smote过采样
smote = SMOTE(random_state=42)
# 将训练集进行SMOTE处理
x_train, y_train= smote.fit_resample(x_train,y_train)
# 使用随机森林进行特征选择
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(x_train, y_train)

# 使用 SelectFromModel 选择重要特征
sfm = SelectFromModel(rf_clf, threshold="mean", max_features=400)
sfm.fit(x_train, y_train)

# 获取选择的特征
selected_features_rf = data.columns[sfm.get_support()]

# 选择的特征
x_train_rf = x_train[selected_features_rf]
x_test_rf = x_test[selected_features_rf]

# 使用CatBoost进行训练
model = CatBoostClassifier(iterations=100, learning_rate=0.11, depth=6)
model.fit(x_train_rf, y_train)
joblib.dump(model, '(CTDC-EGAAC)-CB-R.joblib')
# 预测
y_pred_rf = model.predict(x_test_rf)

# 计算准确率
print(f"准确率为：{accuracy_score(y_pred_rf, y_test)}")

y_score_rf = model.predict_proba(x_test_rf)[:, 1]
# 计算不同阈值下的TPR和FPR
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf, pos_label=1)
# 计算AUC
roc_auc_rf = auc(fpr_rf, tpr_rf)
print(f"AUC = {roc_auc_rf}")
def calculate_sn_sp_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    sn = TP / (TP + FN)
    sp = TN / (TN + FP)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return sn, sp, mcc
# 计算SN、SP和MCC
sn_rf, sp_rf, mcc_rf = calculate_sn_sp_mcc(y_test, y_pred_rf)
print("SN:", sn_rf)
print("SP:", sp_rf)
print("MCC:", mcc_rf)




