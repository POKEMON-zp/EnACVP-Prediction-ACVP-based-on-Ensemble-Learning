import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 假设你的表格文件是CSV格式，文件名为'data.csv'
file_path = r"D:\Python\pythonProject\python\机器学习大模型作业\数据集\原始数据\正.csv"
file_path2=r"D:\Python\pythonProject\python\机器学习大模型作业\数据集\原始数据\负.csv"
# 读取CSV文件到DataFrame
df_p = pd.read_csv(file_path)
df_n = pd.read_csv(file_path2)

P_seq = ''.join(df_p['seq'].astype(str))
N_seq = ''.join(df_n['seq'].astype(str))


def calculate_amino_acid_frequency(sequence):
    # 初始化一个字典来存储每种氨基酸的计数
    amino_acid_counts = {}

    # 遍历序列，统计每种氨基酸的出现次数
    for amino_acid in sequence:
        if amino_acid in amino_acid_counts:
            amino_acid_counts[amino_acid] += 1
        else:
            amino_acid_counts[amino_acid] = 1

    # 计算总长度
    total_length = len(sequence)

    # 计算每种氨基酸的频率
    amino_acid_frequencies = {aa: count / total_length for aa, count in amino_acid_counts.items()}

    return amino_acid_frequencies
# 计算频率
frequencies_p = calculate_amino_acid_frequency(P_seq)
frequencies_n = calculate_amino_acid_frequency(N_seq)
# 将频率转换为列表以便绘图
amino_acids = list(frequencies_p.keys())
frequencies_p_list = [frequencies_p[aa] for aa in amino_acids]
frequencies_n_list = [frequencies_n[aa] for aa in amino_acids]

# 设置图形大小
plt.figure(figsize=(14, 7))
x=np.arange(1,21)
# 绘制正样本频率直方图
plt.bar(x, frequencies_p_list, width=0.4, label='Positive', color='skyblue',tick_label=amino_acids)

# 绘制负样本频率直方图
plt.bar(x+0.4, frequencies_n_list, width=0.4, label='Negative', color='orange')

# 添加标题和标签
plt.title('Amino Acid Frequency Distribution')
plt.xlabel('Amino Acids')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()