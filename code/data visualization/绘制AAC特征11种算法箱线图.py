import matplotlib.pyplot as plt
import numpy as np

# 模型名称和对应的ACC值
models = ['SVM', 'RF', 'Adaboost', 'Catboost', '1DCNN',
          'LSTM', 'BiLSTM', 'Attention+1DCNN', 'Attention+LSTM', 'Attention+BiLSTM']
acc_values = [97.50, 86.33, 91.62, 97.85, 97.26, 97.26, 97.17, 96.36, 96.76, 97.03]

# 生成模拟数据（每个模型5次实验，添加随机噪声）
np.random.seed(42)
data = []
for acc in acc_values:
    noise = np.random.normal(0, 0.5, 5)  # 标准差设为0.5%
    simulated_acc = np.clip(acc + noise, 0, 100)  # 确保ACC在0-100之间
    data.append(simulated_acc)

# 绘制箱线图
plt.figure(figsize=(12, 6))
plt.boxplot(data, labels=models, patch_artist=True)

# 设置样式
plt.xlabel('Models', fontsize=12)
plt.ylabel('ACC (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(80, 100)  # 根据数据范围调整

plt.tight_layout()
plt.show()