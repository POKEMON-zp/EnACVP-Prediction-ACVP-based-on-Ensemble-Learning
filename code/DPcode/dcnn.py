import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten,Attention
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
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
X_train = np.repeat(X_train, 2, axis=1)  # 重复时间步长，使形状变为 (num_samples, 2, 1280)
X_test = np.repeat(X_test, 2, axis=1)    # 重复时间步长，使形状变为 (num_samples, 2, 1280)
# print(X_train.shape[2])
def build_model(input_shape):
    model = Sequential()
    # 添加一个全连接层（Dense），输出维度为128，输入形状由参数input_shape指定
    model.add(Dense(128, input_shape=input_shape))

    # 添加一个一维卷积层（Conv1D），过滤器数量为112，卷积核大小为1，无填充，激活函数为ReLU，权重初始化方式为均匀分布
    model.add(Conv1D(filters=112, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))

    # 添加一个一维最大池化层（MaxPooling1D），池化窗口大小为2，无填充
    model.add(MaxPooling1D(pool_size=2, padding='valid'))

    # 添加另一个一维卷积层，过滤器数量为64，卷积核大小为1，无填充，激活函数为ReLU，权重初始化方式为均匀分布
    model.add(Conv1D(filters=64, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))

    # 添加一个一维最大池化层，池化窗口大小为1，无填充
    model.add(MaxPooling1D(pool_size=1, padding='valid'))

    # 添加一个Dropout层，丢弃率为0.2，防止过拟合
    # model.add(Dropout(0.2))

    # 将多维数据展平成一维
    model.add(Flatten())

    # 添加一个全连接层，输出维度为100，激活函数为ReLU，权重初始化方式为均匀分布
    model.add(Dense(100, activation='relu', kernel_initializer='uniform'))

    # 添加一个全连接层，输出维度为1，激活函数为Sigmoid，用于二分类任务
    model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

    # 编译模型，损失函数为二元交叉熵（binary_crossentropy），优化器为Adam，评估指标为准确率（accuracy）
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

input_shape = (2,X_train.shape[2])
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

#ESM
#无dropout
# SN: 0.6727272727272727
# SP: 0.9644549763033176
# MCC: 0.6529878520179063
# 测试集上的准确率: 0.9308176100628931
# 测试集上的AUC: 0.9155536406721241

#AAC+EAAC
#dropout = 0.2
# SN: 0.6666666666666666
# SP: 0.9339622641509434
# MCC: 0.5519546456498513
# 测试集上的准确率: 0.9052631578947369
# 测试集上的AUC: 0.8633694043655197

#CTDC+EGAAC
#dropout = 0.2
# SN: 0.5490196078431373
# SP: 0.9551886792452831
# MCC: 0.522765817481095
# 测试集上的准确率: 0.911578947368421
# 测试集上的AUC: 0.8349056603773585