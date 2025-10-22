import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取第一个CSV文件
df1 = pd.read_csv(r"D:\Python\pythonProject\python\机器学习大模型作业\数据集\原始数据\正.csv")

# 读取第二个CSV文件
df2 = pd.read_csv(r"D:\Python\pythonProject\python\机器学习大模型作业\数据集\原始数据\负.csv")

# 计算每个蛋白质序列的长度
df1['Sequence_Length'] = df1['seq'].apply(len)
df2['Sequence_Length'] = df2['seq'].apply(len)

custom_palette = ["#FF5733", "#33FF57"]
# 绘制小提琴图
plt.figure(figsize=(10, 6))
plt.subplot(121)
sns.violinplot(x=df1['label'], y=df1['Sequence_Length'], palette=custom_palette)
plt.xlabel('ACVPS')
plt.ylabel('Sequence Length')
plt.subplot(122)
sns.violinplot(x=df2['label'], y=df2['Sequence_Length'],  palette='muted')
plt.xlabel('non-ACVPS')
plt.ylabel('Sequence Length')
plt.show()