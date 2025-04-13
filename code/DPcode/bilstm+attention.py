import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, Dropout
from keras import backend as K
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(1,),
                                 initializer='zeros',
                                 trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        # 计算注意力得分 (batch_size, timesteps, 1)
        e = K.tanh(K.dot(x, self.W) + self.b)
        # 获取注意力权重 (batch_size, timesteps)
        a = K.softmax(K.squeeze(e, axis=-1), axis=-1)
        # 应用注意力权重 (batch_size, timesteps, features)
        weighted_input = x * K.expand_dims(a)
        # 输出加权和 (batch_size, features)
        return K.sum(weighted_input, axis=1)


# 读取数据
df = pd.read_csv("CTDC.csv")
data = df.iloc[:, 1:].values
target = df.iloc[:, 0].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42, stratify=target
)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用SMOTE过采样
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 调整数据维度 (samples, timesteps, features)
timesteps = 10  # 根据数据特性调整时间步长
features = X_train.shape[1] // timesteps
X_train = X_train.reshape((-1, timesteps, features))
X_test = X_test.reshape((-1, timesteps, features))


def build_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        SelfAttention(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


# 训练参数
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

model = build_model((timesteps, features))
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 评估模型
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)


def calculate_sn_sp_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    sn = TP / (TP + FN) if (TP + FN) != 0 else 0
    sp = TN / (TN + FP) if (TN + FP) != 0 else 0
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if (TP + FP) * (TP + FN) * (
                TN + FP) * (TN + FN) != 0 else 0

    return sn, sp, mcc


# 修改后的评估函数
def enhanced_evaluation(y_true, y_pred):
    # 保持原有指标计算
    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    sn, sp, mcc = calculate_sn_sp_mcc(y_true, y_pred)

    # 打印原有指标
    print("\n=== 核心评价指标 ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"SN (Recall): {sn:.4f}")
    print(f"SP: {sp:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC: {roc_auc:.4f}")

# 调用评估函数
enhanced_evaluation(y_test, y_pred_classes)
