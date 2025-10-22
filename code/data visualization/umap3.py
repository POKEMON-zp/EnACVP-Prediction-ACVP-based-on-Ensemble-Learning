import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from umap import UMAP

# 设置显示选项
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# 读取数据
df = pd.read_csv("ESM.csv")

# 分离特征和目标变量
data = df.iloc[:, 1:]
target = df.iloc[:, 0]
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=66)

# UMAP降维
umap = UMAP(n_neighbors=100, n_components=2, random_state=42)
X_umap = umap.fit_transform(x_train)

# 将目标变量映射为标签
y_train_labels = y_train.map({0: 'non-ACVP', 1: 'ACVP'})

# 创建颜色映射
color_map = {'non-ACVP': 'blue', 'ACVP': 'orange'}
colors = y_train_labels.map(color_map)

# 绘图
plt.figure(figsize=(10, 8))
for label, color in color_map.items():
    idx = y_train_labels == label
    plt.scatter(X_umap[idx, 0], X_umap[idx, 1], c=color, label=label, alpha=0.7, edgecolors='k', s=50)

# 图例设置
plt.legend(loc='upper right')
plt.title('UMAP of ESM2 CB Training Data')
plt.grid(False)
plt.show()
