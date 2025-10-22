import pandas as pd
import plotly.graph_objects as go

# 读取数据
data = {
    '模型组合': ['Model1+2+4+5+7', 'Model3+4+5+7', 'Model1+2+6+7', 'Model3+6+7', 'Model1+7', 'Model2+7', 'Model3+7', 'Model4+7', 'Model5+7', 'Model6+7'],
    'ACC(%)': [94.53, 93.26, 94.11, 93.89, 94.32, 92.00, 94.74, 97.47, 91.58, 94.74],
    'SN(%)': [55.56, 44.44, 50.00, 50.00, 50.00, 29.63, 53.70, 79.62, 25.93, 53.70],
    'SP(%)': [99.52, 99.52, 99.76, 99.52, 100.00, 100.00, 100.00, 99.76, 100.00, 100.00],
    'MCC': [69.75, 61.36, 67.07, 65.66, 68.55, 52.13, 71.20, 86.92, 48.66, 71.20],
    'AUC(%)': [96.04, 90.76, 90.36, 92.32, 90.83, 86.87, 91.81, 97.57, 85.07, 89.65]
}
df = pd.DataFrame(data)

# 准备Sankey图数据
nodes = list(df['模型组合']) + ['ACC', 'SN', 'SP', 'MCC', 'AUC']
source = []
target = []
value = []

for index, row in df.iterrows():
    model = row['模型组合']
    for metric in ['ACC(%)', 'SN(%)', 'SP(%)', 'MCC', 'AUC(%)']:
        source.append(nodes.index(model))
        target.append(nodes.index(metric.replace('(%)', '')))
        value.append(row[metric])

# 创建Sankey图
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=nodes
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])

# 更新图表布局
fig.update_layout(title='算法集成结果类似弦图（Sankey图替代）', font=dict(size=12))

# 显示图表
fig.show()