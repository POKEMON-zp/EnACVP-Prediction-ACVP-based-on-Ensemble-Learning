from sklearn.model_selection import train_test_split
import shap
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import joblib

# 读取数据
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
df = pd.read_csv("AAC.csv")
data = df.iloc[:, 1:]
target = df.iloc[:, 0]

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=66)

# 使用smote过采样
smote = SMOTE(random_state=42)
# 将训练集进行SMOTE处理
x_train, y_train = smote.fit_resample(x_train, y_train)

# 加载模型
model = joblib.load('AAC-CB.joblib')

# SHAP可视化
# 创建解释器
explainer = shap.Explainer(model, x_train)
shap_values = explainer(x_test)

# 绘制Shap图并设置纵坐标名称为 "Feature Importance"
feature_names =['AAC1', 'AAC2', 'AAC3', 'AAC4', 'AAC5', 'AAC6', 'AAC7', 'AAC8', 'AAC9', 'AAC10', 'AAC11', 'AAC12', 'AAC13', 'AAC14', 'AAC15', 'AAC16', 'AAC17', 'AAC18', 'AAC19', 'AAC20']



shap.summary_plot(shap_values, x_test, feature_names=feature_names)

# Feature_name_list = [f"EGAAC{i}" for i in range(1, 461)]
#
# # 打印整个列表（注意：这里打印整个列表会非常长，可能会占用大量终端空间）
# print(Feature_name_list)
