import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM,Bidirectional
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取数据
df = pd.read_csv("EGAAC.csv")
data = df.iloc[:, 1:]
target = df.iloc[:, 0]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 使用 SMOTE 过采样
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 重塑输入数据为适合LSTM的格式 (样本数, 时间步长, 特征数)
timesteps = 1  # 定义时间步长
features = X_train.shape[1] // timesteps  # 计算每个时间步长的特征数
X_train = np.reshape(X_train, (X_train.shape[0], timesteps, features))
X_test = np.reshape(X_test, (X_test.shape[0], timesteps, features))


def build_model(input_shape):
    model = Sequential()
    # 添加一个LSTM层，输出维度为64，输入形状由参数input_shape指定
    model.add(Bidirectional(LSTM(64, input_shape=input_shape)))

    # 添加一个Dropout层，丢弃率为0.2，防止过拟合
    # model.add(Dropout(0.2))

    # 添加一个全连接层，输出维度为100，激活函数为ReLU，权重初始化方式为均匀分布
    model.add(Dense(100, activation='relu', kernel_initializer='uniform'))

    # 添加一个全连接层，输出维度为1，激活函数为Sigmoid，用于二分类任务
    model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

    # 编译模型，损失函数为二元交叉熵（binary_crossentropy），优化器为Adam，评估指标为准确率（accuracy）
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

input_shape = (timesteps, features)
model = build_model(input_shape)

# 训练模型
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# 在测试集上评估模型
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# 计算准确率和其他评估指标
accuracy = accuracy_score(y_test, y_pred_classes)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)


def calculate_sn_sp_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    sn = TP / (TP + FN)
    sp = TN / (TN + FP)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return sn, sp, mcc


sn, sp, mcc = calculate_sn_sp_mcc(y_test, y_pred_classes)
print("SN:", sn)
print("SP:", sp)
print("MCC:", mcc)

print(f"测试集上的准确率: {accuracy}")
print(f"测试集上的AUC: {roc_auc}")
