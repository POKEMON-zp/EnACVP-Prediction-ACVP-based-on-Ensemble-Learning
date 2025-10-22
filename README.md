# 
采用的正样本来源于APD3、DBASSP、AVPDB等数据库，负样本来源于Uniport中去除包含抗菌肽、抗病毒肽、抗癌肽等关键词后得到，并对经过去除冗余以及等长序列处理后的正负样本划分进行数据分析，并引入包含蛋白质语言模型ESM2在内的三类特征进行特征表示，并利用SMOTE进行数据的平衡化处理，分别对特征及特征组合利用基于LGBM的特征选择后，从11种机器学习和深度学习模型中筛选出适合每类特征的最佳模型
<img width="1174" height="639" alt="image" src="https://github.com/user-attachments/assets/c0cbeedf-e055-4451-982e-690ba326d12d" />
