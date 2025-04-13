import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import tensorflow as tf


# 改进的自注意力层（修正维度处理）
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 修正权重维度为 (features, 1)
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(1,),
                                 initializer='zeros',
                                 trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        # 计算注意力得分 (batch_size, timesteps, 1)
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        # 在时间步维度应用softmax
        a = tf.nn.softmax(e, axis=1)
        # 应用注意力权重 (batch_size, timesteps, features)
        weighted = x * a
        # 沿时间步维度求和 (batch_size, features)
        return tf.reduce_sum(weighted, axis=1)


# 数据准备
df = pd.read_csv("CTDC.csv")
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# 数据标准化（重要改进）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分层划分数据集（保持类别分布）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE过采样
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 调整数据维度（增加时间步长维度）
timesteps = 10  # 将特征划分为10个时间步
features = X_train.shape[1] // timesteps
X_train = X_train.reshape(-1, timesteps, features)
X_test = X_test.reshape(-1, timesteps, features)


# 优化模型结构
def create_optimized_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape,
             kernel_initializer='he_normal', recurrent_dropout=0.2),
        SelfAttention(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


# 训练配置
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

model = create_optimized_model((timesteps, features))

# 训练模型（增加验证集）
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)


# 评价指标计算（保持原有逻辑不变）
def calculate_sn_sp_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    sn = TP / (TP + FN) if (TP + FN) != 0 else 0
    sp = TN / (TN + FP) if (TN + FP) != 0 else 0
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / denominator if denominator != 0 else 0

    return sn, sp, mcc


# 评估模型
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred_classes)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
sn, sp, mcc = calculate_sn_sp_mcc(y_test, y_pred_classes)

# 输出结果（保持原有格式）
print("\n=== 核心评价指标 ===")
print(f"SN (敏感性): {sn:.4f}")
print(f"SP (特异性): {sp:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"准确率: {accuracy:.4f}")
print(f"AUC值: {roc_auc:.4f}")