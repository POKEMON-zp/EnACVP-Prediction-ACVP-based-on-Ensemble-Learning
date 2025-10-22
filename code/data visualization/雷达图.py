import numpy as np
import matplotlib.pyplot as plt

# 数据准备
features = ['ACC', 'SN', 'SP', 'MCC', 'AUC']
smote_data = [97.47	,79.62,	99.76,	86.92,	97.57]
random_data = [92.84,	37.04,	100.00,	58.54	,95.22]

# 角度计算与数据闭合
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]  # 闭合角度
smote_data += smote_data[:1]  # 闭合数据
random_data += random_data[:1]

# 创建极坐标画布
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# 绘制雷达图
ax.plot(angles, smote_data, 'o-', linewidth=2, label='SMOTE', color='#1f77b4')
ax.fill(angles, smote_data, color='#1f77b4', alpha=0.1)
ax.plot(angles, random_data, 'o-', linewidth=2, label='Random', color='#ff7f0e')
ax.fill(angles, random_data, color='#ff7f0e', alpha=0.1)

# 设置极坐标参数
ax.set_theta_offset(np.pi / 2)    # 0度方向朝上
ax.set_theta_direction(-1)        # 顺时针方向
ax.set_rlabel_position(0)         # 径向标签位置
ax.set_thetagrids(np.degrees(angles[:-1]), labels=features)  # 特征标签

# 设置径向轴范围和刻度
plt.ylim(0, 100)
plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)

# 添加标题和图例
plt.title('ensem', fontweight='bold',fontsize = 20,y = 1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.1))

# 显示图形
plt.tight_layout()
plt.show()