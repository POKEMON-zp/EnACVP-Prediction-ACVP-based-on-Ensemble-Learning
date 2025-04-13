import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Attention, Input
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
X_test = np.repeat(X_test, 2, axis=1)  # 重复时间步长，使形状变为 (num_samples, 2, 1280)


def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=112, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform')(inputs)
    x = MaxPooling1D(pool_size=2, padding='valid')(x)
    x = Conv1D(filters=64, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform')(x)
    x = MaxPooling1D(pool_size=1, padding='valid')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer='uniform')(x)

    # 添加自注意力层
    attention_output = Attention()([x, x])

    outputs = Dense(1, activation='sigmoid', kernel_initializer='uniform')(attention_output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


input_shape = (2, X_train.shape[2])
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
# SN: 0.6545454545454545
# SP: 0.9881516587677726
# MCC: 0.7323332394179274
# 测试集上的准确率: 0.949685534591195
# 测试集上的AUC: 0.8707453683757002

#AAC+EAAC
#无dropout
# SN: 0.43137254901960786
# SP: 0.9858490566037735
# MCC: 0.5484057872352973
# 测试集上的准确率: 0.9263157894736842
# 测试集上的AUC: 0.8260960044395116

#CTDC+EGAAC
#无dropout
# SN: 0.49019607843137253
# SP: 0.9622641509433962
# MCC: 0.49878299029362033
# 测试集上的准确率: 0.911578947368421
# 测试集上的AUC: 0.7982796892341842