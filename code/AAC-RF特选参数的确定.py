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
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("AAC.csv")
data = df.iloc[:, 1:]
target = df.iloc[:, 0]

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=66)

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

# LightGBM特征选择
lgb_clf = lgb.LGBMClassifier(random_state=42)
lgb_clf.fit(x_train, y_train)
feature_importances = lgb_clf.feature_importances_

# 用于存储不同特征数量对应的准确率
num_features = []
accuracies = []

for num in range(1, 21):
    selected_features = data.columns[np.argsort(feature_importances)[::-1][:num]]  # 选择前num个重要特征
    x_train_selected = x_train[selected_features]
    x_test_selected = x_test[selected_features]

    # 创建模型
    model = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=1, min_samples_split=2)
    model.fit(x_train_selected, y_train)
    # 预测
    y_pred = model.predict(x_test_selected)
    # 计算准确率
    accuracy = accuracy_score(y_pred, y_test)
    num_features.append(num)
    accuracies.append(accuracy)

# 绘制折线图
plt.plot(num_features, accuracies, marker='o', linestyle='-', color='blue')
plt.title('Feature Selection Accuracy vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.grid(True)

# 设置横纵坐标取值范围
plt.xlim(0, 21)
plt.ylim(min(accuracies) - 0.01, max(accuracies) + 0.01)
plt.xticks(range(0, 22))

plt.show()
