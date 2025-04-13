from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import precision_score,recall_score,f1_score,roc_curve,auc,confusion_matrix,accuracy_score,matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier
import xgboost
import matplotlib.pyplot as plt
import csv
import pickle
from joblib import dump, load
#输出结果无省略号
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 100000)

#读入文件
#AAC
f1 = open("AAC_test.csv", 'r')
csvreader1 = csv.reader(f1)
final_list1 = list(csvreader1)
data1 = []
target1 = []
for file in final_list1:
    data1.append(file[1:])
    target1.append(int(file[0]))
#CTDC
#读入文件
f2 = open("CTDC_test.csv", 'r')
csvreader2 = csv.reader(f2)
final_list2 = list(csvreader2)
data2 = []
target2 = []
for file in final_list2:
    data2.append(file[1:])
    target2.append(int(file[0]))
#EAAC
#读入文件
f3 = open("EAAC_test.csv", 'r')
csvreader3 = csv.reader(f3)
final_list3 = list(csvreader3)
data3 = []
target3 = []
for file in final_list3:
    data3.append(file[1:])
    target3.append(int(file[0]))
#EGAAC
#读入文件
f = open("EGAAC_test.csv", 'r')
csvreader4 = csv.reader(f)
final_list4 = list(csvreader4)
data4 = []
target4 = []
for file in final_list4:
    data4.append(file[1:])
    target4.append(int(file[0]))
#esm2
#读入文件
f = open("ESM_test.csv", 'r')
csvreader5 = csv.reader(f)
final_list5 = list(csvreader5)
data5 = []
target5 = []
for file in final_list5:
    data5.append(file[1:])
    target5.append(int(file[0]))

RF_AAC=load('AAC-RF.joblib')
RF_CTDC=load('CTDC-RF.joblib')
RF_EAAC=load('EAAC-RF.joblib')
RF_EGAAC=load('EGAAC-RF.joblib')
RF_ESM=load('ESM-RF.joblib')
CB_CTDC=load('CTDC-CB.joblib')
CB_AAC=load('AAC-CB.joblib')
CB_EAAC=load('EAAC-CB.joblib')
CB_EGAAC=load('EGAAC-CB.joblib')
CB_ESM=load('ESM-CB.joblib')
SVM_AAC=load('AAC-SVM.joblib')
SVM_CTDC=load('CTDC-SVM.joblib')
SVM_EAAC=load('EAAC-SVM.joblib')
SVM_EGAAC=load('EGAAC-SVM.joblib')
SVM_ESM=load('ESM-SVM.joblib')


AAC_RF_predict_proba=RF_AAC.predict_proba(data1)
AAC_CB_predict_proba=CB_AAC.predict_proba(data1)
AAC_SVM_predict_proba=SVM_AAC.predict_proba(data1)

CTDC_RF_predict_proba=RF_CTDC.predict_proba(data2)
CTDC_CB_predict_proba=CB_CTDC.predict_proba(data2)
CTDC_SVM_predict_proba=SVM_CTDC.predict_proba(data2)

# CTDC_RF_predict_proba=CB_CTDC.predict_proba(data2)
# CTDC_CB_predict_proba=SVM_CTDC.predict_proba(data2)
# CTDC_SVM_predict_proba=CB_CTDC.predict_proba(data2)

EAAC_RF_predict_proba=RF_EAAC.predict_proba(data3)
EAAC_CB_predict_proba=CB_EAAC.predict_proba(data3)
EAAC_SVM_predict_proba=SVM_EAAC.predict_proba(data3)

EGAAC_RF_predict_proba=RF_EGAAC.predict_proba(data4)
EGAAC_CB_predict_proba=CB_EGAAC.predict_proba(data4)
EGAAC_SVM_predict_proba=SVM_EGAAC.predict_proba(data4)

ESM_RF_predict_proba=RF_ESM.predict_proba(data5)
ESM_CB_predict_proba=CB_ESM.predict_proba(data5)
ESM_SVM_predict_proba=SVM_ESM.predict_proba(data5)

# 5个
# def Final_predictor1(AAC_predict_proba,EAAC_predict_proba,CTDC_predict_proba,EGAAC_predict_proba,ESM_predict_proba):
#     Final_predict=[]
#     Final_predict_proba=[]
#     num=int(len(target1))
#     for i in range(num):
#         a,b,c,d,e=AAC_predict_proba.tolist()[i][1],EAAC_predict_proba.tolist()[i][1],CTDC_predict_proba.tolist()[i][1],EGAAC_predict_proba.tolist()[i][1],ESM_predict_proba.tolist()[i][1]
#         avg=(a+b+c+d+e)/5
#         thre=0.5
#         Final_predict_proba.append(avg)
#         if avg>=thre:
#             Final_predict.append(1)
#         else:
#             Final_predict.append(0)
#     return np.array(Final_predict),np.array(Final_predict_proba)
# Final_predict,Final_predict_proba=Final_predictor1(AAC_CB_predict_proba,CTDC_CB_predict_proba,EAAC_SVM_predict_proba,EGAAC_SVM_predict_proba,ESM_CB_predict_proba)
#
# #评价指标
# # 计算SN和SP
# tn, fp, fn, tp = confusion_matrix(target1, Final_predict).ravel()
# SN = tp / (tp + fn)
# SP = tn / (tn + fp)
# fpr1, tpr1, thread1 = roc_curve(target1, Final_predict_proba)
# roc_auc1 = auc(fpr1, tpr1)
# print("ACC:{:.4f}".format(accuracy_score(target1, Final_predict)))
# print("AUC:{:.4f}".format(roc_auc1))
# print("SN:{:.4f}".format(SN))
# print("SP:{:.4f}".format(SP))
# print("MCC:{:.4f}".format(matthews_corrcoef(target1, Final_predict)))


#4个
# def Final_predictor2(AAC_predict_proba,EAAC_predict_proba,CTDC_predict_proba,EGAAC_predict_proba):
#     Final_predict=[]
#     Final_predict_proba=[]
#     num=int(len(target1))
#     for i in range(num):
#         a,b,c,d=AAC_predict_proba.tolist()[i][1],EAAC_predict_proba.tolist()[i][1],CTDC_predict_proba.tolist()[i][1],EGAAC_predict_proba.tolist()[i][1]
#         avg=(a+b+c+d)/4
#         thre=0.5
#         Final_predict_proba.append(avg)
#         if avg>=thre:
#             Final_predict.append(1)
#         else:
#             Final_predict.append(0)
#     return np.array(Final_predict),np.array(Final_predict_proba)
# Final_predict1,Final_predict_proba1=Final_predictor2(CTDC_SVM_predict_proba,EAAC_CB_predict_proba,AAC_CB_predict_proba,ESM_CB_predict_proba)

#评价指标
# 计算SN和SP
# tn, fp, fn, tp = confusion_matrix(target1, Final_predict).ravel()
# SN = tp / (tp + fn)
# SP = tn / (tn + fp)
# fpr1, tpr1, thread1 = roc_curve(target1, Final_predict_proba)
# roc_auc1 = auc(fpr1, tpr1)
# print("ACC:{:.4f}".format(accuracy_score(target1, Final_predict)))
# print("AUC:{:.4f}".format(roc_auc1))
# print("SN:{:.4f}".format(SN))
# print("SP:{:.4f}".format(SP))
# print("MCC:{:.4f}".format(matthews_corrcoef(target1, Final_predict)))

#3个
# def Final_predictor3(AAC_predict_proba,EAAC_predict_proba,CTDC_predict_proba):
#     Final_predict=[]
#     Final_predict_proba=[]
#     num=int(len(target1))
#     for i in range(num):
#         a,b,c=AAC_predict_proba.tolist()[i][1],EAAC_predict_proba.tolist()[i][1],CTDC_predict_proba.tolist()[i][1]
#         avg=(a+b+c)/3
#         thre=0.5
#         Final_predict_proba.append(avg)
#         if avg>=thre:
#             Final_predict.append(1)
#         else:
#             Final_predict.append(0)
#     return np.array(Final_predict),np.array(Final_predict_proba)
# Final_predict2,Final_predict_proba2=Final_predictor3(EGAAC_CB_predict_proba,CTDC_CB_predict_proba,ESM_CB_predict_proba)
# #
# #评价指标
# # 计算SN和SP
# tn, fp, fn, tp = confusion_matrix(target1, Final_predict2).ravel()
# SN = tp / (tp + fn)
# SP = tn / (tn + fp)
# fpr1, tpr1, thread1 = roc_curve(target1, Final_predict_proba2)
# roc_auc1 = auc(fpr1, tpr1)
# print("ACC:{:.4f}".format(accuracy_score(target1, Final_predict2)))
# print("AUC:{:.4f}".format(roc_auc1))
# print("SN:{:.4f}".format(SN))
# print("SP:{:.4f}".format(SP))
# print("MCC:{:.4f}".format(matthews_corrcoef(target1, Final_predict2)))

#2个
def Final_predictor4(AAC_predict_proba,EAAC_predict_proba):
    Final_predict=[]
    Final_predict_proba=[]
    num=int(len(target1))
    for i in range(num):
        a,b=AAC_predict_proba.tolist()[i][1],EAAC_predict_proba.tolist()[i][1]
        avg=(a+b)/2
        thre=0.5
        Final_predict_proba.append(avg)
        if avg>=thre:
            Final_predict.append(1)
        else:
            Final_predict.append(0)
    return np.array(Final_predict),np.array(Final_predict_proba)
Final_predict3,Final_predict_proba3=Final_predictor4(ESM_CB_predict_proba,CTDC_CB_predict_proba)
# Final_predict4,Final_predict_proba4=Final_predictor4(AAC_CB_predict_proba,CTDC_CB_predict_proba)
# Final_predict5,Final_predict_proba5=Final_predictor4(ESM_CB_predict_proba,EGAAC_CB_predict_proba)
# #评价指标
# 计算SN和SP
tn, fp, fn, tp = confusion_matrix(target1, Final_predict3).ravel()
SN = tp / (tp + fn)
SP = tn / (tn + fp)
fpr1, tpr1, thread1 = roc_curve(target1, Final_predict_proba3)
roc_auc1 = auc(fpr1, tpr1)
print("ACC:{:.4f}".format(accuracy_score(target1, Final_predict3)))
print("AUC:{:.4f}".format(roc_auc1))
print("SN:{:.4f}".format(SN))
print("SP:{:.4f}".format(SP))
print("MCC:{:.4f}".format(matthews_corrcoef(target1, Final_predict3)))

# #ROC曲线
# names = ['AAC(C)_CTDC(S)_EAAC(C)_ESM(C)',
#          'ACC(C)_EAAC(C)_ESM(SVM)',
#          'ESM(C)_CTDC(C)',
#          'ACC(C)_CTDC(C)',
#          'ESM(C)_EGAAC(C)',
#          'AAC(C)_EAAC(R)_ESM(C)']
# y_test_predprob=[Final_predict_proba1,
#                  Final_predict_proba2,
#                  Final_predict_proba3,
#                  Final_predict_proba4,
#                  Final_predict_proba5,
#                  Final_predict_proba6]
# colors = ['crimson',
#           'yellow',
#           'mediumseagreen',
#           'steelblue',
#           'purple',
#           'orange'
# ]
#
# def multi_models_roc(names, y_test_predprob ,colors,save=True, dpin=100):
#     plt.figure(figsize=(20, 20), dpi=dpin)
#
#     for (name, predict_Y, colorname) in zip(names, y_test_predprob, colors):
#         fpr, tpr, thresholds = roc_curve(target1, predict_Y, pos_label=1)
#
#         plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
#         plt.plot([0, 1], [0, 1], '--', lw=5, color='grey')
#         plt.axis('square')
#         plt.xlim([0, 1])
#         plt.ylim([0, 1])
#         plt.xlabel('False Positive Rate', fontsize=20)
#         plt.ylabel('True Positive Rate', fontsize=20)
#         plt.title('Different feature conbination_ROC Curve', fontsize=20)
#         plt.legend(loc='lower right', fontsize=15)
#
#     return plt.show()
#
# train_roc_graph = multi_models_roc(names, y_test_predprob, colors)