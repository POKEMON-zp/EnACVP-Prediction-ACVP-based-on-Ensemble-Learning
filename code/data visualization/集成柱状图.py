import pandas as pd
import matplotlib.pyplot as plt

# 数据
data = {
    'Model': ['Model1+2+4+5+7', 'Model3+4+5+7', 'Model1+2+6+7', 'Model3+6+7', 'Model1+7', 'Model2+7', 'Model3+7',
              'Model4+7', 'Model5+7', 'Model6+7'],
    'ACC(%)': [94.53, 93.26, 94.11, 93.89, 94.32, 92.00, 94.74, 97.47, 91.58, 94.74],
    'SN(%)': [55.56, 44.44, 50.00, 50.00, 50.00, 29.63, 53.70, 79.62, 25.93, 53.70],
    'SP(%)': [99.52, 99.52, 99.76, 99.52, 100.00, 100.00, 100.00, 99.76, 100.00, 100.00],
    'MCC': [69.75, 61.36, 67.07, 65.66, 68.55, 52.13, 71.20, 86.92, 48.66, 71.20],
    'AUC(%)': [96.04, 90.76, 90.36, 92.32, 90.83, 86.87, 91.81, 97.57, 85.07, 89.65]
}

df = pd.DataFrame(data)

# 设置中文字体为黑体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 指标列表
metrics = ['ACC(%)', 'SN(%)', 'SP(%)', 'MCC', 'AUC(%)']
num_models = len(df['Model'])
# 减少柱子宽度，增加间隔
bar_width = 0.06
# 指标间的间隔
gap = 0.3
bar_positions = [list(range(len(metrics))) for _ in range(num_models)]
for i in range(1, num_models):
    for j in range(len(metrics)):
        bar_positions[i][j] = bar_positions[i - 1][j] + bar_width
    # 应用指标间的间隔
    for j in range(1, len(metrics)):
        for k in range(num_models):
            bar_positions[k][j] = bar_positions[k][j - 1] + bar_width * num_models + gap

# 绘制柱状图
fig, ax = plt.subplots(figsize=(15, 8))  # 增加图形宽度
for i, model in enumerate(df['Model']):
    ax.bar(bar_positions[i], df.loc[i, metrics], width=bar_width, label=model)

# 设置轴标签和标题
ax.set_xlabel('Evaluation metrics')
ax.set_ylabel('Metric')
ax.set_title('Comparison of indicators for different model combinations')

# 设置 x 轴刻度标签
ax.set_xticks([bar_positions[0][i] + bar_width * (num_models - 1) / 2 for i in range(len(metrics))])
ax.set_xticklabels(metrics)

# 显示图例，调整位置到图形外部右上角
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.5)

# 显示图形
plt.tight_layout()  # 自动调整布局
plt.show()
