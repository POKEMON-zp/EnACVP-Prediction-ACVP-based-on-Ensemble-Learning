import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from umap import UMAP
from matplotlib.colors import Normalize

# 设置显示选项
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# 读取数据
df = pd.read_csv("AAC.csv")

# 分离特征和目标变量
data = df.iloc[:, 1:]
target = df.iloc[:, 0]
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=66)

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

# UMAP降维并可视化
umap = UMAP(n_neighbors=550, n_components=2, random_state=42)
X_umap = umap.fit_transform(x_train)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_train, cmap='plasma', alpha=0.7)
plt.colorbar()
plt.title('UMAP of AAC CB Training Data')

# 创建归一化对象
norm = Normalize(vmin=np.min(y_train), vmax=np.max(y_train))

# 添加颜色标签说明
unique_labels = np.unique(y_train)
handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Label {label}',
                      markerfacecolor=plt.cm.plasma(norm(label)), markersize=10) for label in unique_labels]
plt.legend(handles=handles, loc='upper right')
plt.show()
