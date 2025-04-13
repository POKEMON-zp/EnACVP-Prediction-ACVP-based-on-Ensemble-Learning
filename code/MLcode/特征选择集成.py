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

f = open("AAC_SVM_test.csv", 'r')
csvreader6 = csv.reader(f)
final_list6 = list(csvreader6)
data6 = []
target6 = []
for file in final_list6:
    data6.append(file[1:])
    target6.append(int(file[0]))

f = open("CTDC_CB_test.csv", 'r')
csvreader7 = csv.reader(f)
final_list7 = list(csvreader7)
data7 = []
target7 = []
for file in final_list7:
    data7.append(file[1:])
    target7.append(int(file[0]))

f = open("CTDC_SVM_test.csv", 'r')
csvreader8 = csv.reader(f)
final_list8 = list(csvreader8)
data8 = []
target8 = []
for file in final_list8:
    data8.append(file[1:])
    target8.append(int(file[0]))

f = open("CTDC_RF_test.csv", 'r')
csvreader9 = csv.reader(f)
final_list9 = list(csvreader9)
data9 = []
target9 = []
for file in final_list9:
    data9.append(file[1:])
    target9.append(int(file[0]))

f = open("EAAC_SVM_test.csv", 'r')
csvreader10 = csv.reader(f)
final_list10 = list(csvreader10)
data10 = []
target10 = []
for file in final_list10:
    data10.append(file[1:])
    target10.append(int(file[0]))

f = open("EAAC_RF_test.csv", 'r')
csvreader11 = csv.reader(f)
final_list11 = list(csvreader11)
data11 = []
target11 = []
for file in final_list11:
    data11.append(file[1:])
    target11.append(int(file[0]))

f = open("EGAAC_SVM_test.csv", 'r')
csvreader12 = csv.reader(f)
final_list12 = list(csvreader12)
data12 = []
target12 = []
for file in final_list12:
    data12.append(file[1:])
    target12.append(int(file[0]))

f = open("EGAAC_RF_test.csv", 'r')
csvreader13 = csv.reader(f)
final_list13 = list(csvreader13)
data13 = []
target13 = []
for file in final_list13:
    data13.append(file[1:])
    target13.append(int(file[0]))

f = open("ESM_SVM_test.csv", 'r')
csvreader14 = csv.reader(f)
final_list14 = list(csvreader14)
data14 = []
target14 = []
for file in final_list14:
    data14.append(file[1:])
    target14.append(int(file[0]))

f = open("ESM_RF_test.csv", 'r')
csvreader15 = csv.reader(f)
final_list15 = list(csvreader15)
data15 = []
target15 = []
for file in final_list15:
    data15.append(file[1:])
    target15.append(int(file[0]))

f = open("AAC+EAAC.csv", 'r')
csvreader16 = csv.reader(f)
final_list16 = list(csvreader16)
data16 = []
target16 = []
for file in final_list16:
    data16.append(file[1:])
    target16.append(int(file[0]))

f = open("CTDC+EGAAC.csv", 'r')
csvreader17 = csv.reader(f)
final_list17 = list(csvreader17)
data17 = []
target17 = []
for file in final_list17:
    data17.append(file[1:])
    target17.append(int(file[0]))

f = open("ESM.csv", 'r')
csvreader18 = csv.reader(f)
final_list18 = list(csvreader18)
data18 = []
target18 = []
for file in final_list18:
    data18.append(file[1:])
    target18.append(int(file[0]))

RF_AAC=load('AAC-RF.joblib')
RF_CTDC_S=load('CTDC-RF-S.joblib')
RF_EAAC_S=load('EAAC-RF-S.joblib')
RF_EGAAC_S=load('EGAAC-RF-S.joblib')
RF_ESM_S=load('ESM-RF-S.joblib')
CB_CTDC_S=load('CTDC-CB-S.joblib')
CB_AAC=load('AAC-CB.joblib')
CB_EAAC=load('EAAC-CB.joblib')
CB_EGAAC=load('EGAAC-CB.joblib')
CB_ESM=load('ESM-CB.joblib')
SVM_AAC_S=load('AAC-SVM-S.joblib')
SVM_CTDC_S=load('CTDC-SVM-S.joblib')
SVM_EAAC_S=load('EAAC-SVM-S.joblib')
SVM_EGAAC_S=load('EGAAC-SVM-S.joblib')
SVM_ESM_S=load('ESM-SVM-S.joblib')


CB_AAC_EAAC=load('(AAC+EAAC)-CB.joblib')
CB_CTDC_EGAAC = load('(CTDC+EGAAC)-CB.joblib')


AAC_RF_predict_proba=RF_AAC.predict_proba(data1)
AAC_CB_predict_proba=CB_AAC.predict_proba(data1)
AAC_SVM_predict_proba=SVM_AAC_S.predict_proba(data6)

CTDC_RF_predict_proba=RF_CTDC_S.predict_proba(data9)
CTDC_CB_predict_proba=CB_CTDC_S.predict_proba(data2)
CTDC_SVM_predict_proba=SVM_CTDC_S.predict_proba(data8)

EAAC_RF_predict_proba=RF_EAAC_S.predict_proba(data11)
EAAC_CB_predict_proba=CB_EAAC.predict_proba(data3)
EAAC_SVM_predict_proba=SVM_EAAC_S.predict_proba(data10)

EGAAC_RF_predict_proba=RF_EGAAC_S.predict_proba(data13)
EGAAC_CB_predict_proba=CB_EGAAC.predict_proba(data4)
EGAAC_SVM_predict_proba=SVM_EGAAC_S.predict_proba(data12)

ESM_RF_predict_proba=RF_ESM_S.predict_proba(data15)
ESM_CB_predict_proba=CB_ESM.predict_proba(data5)
ESM_SVM_predict_proba=SVM_ESM_S.predict_proba(data14)


AAC_EAAC_CB_predict_proba = CB_AAC_EAAC.predict_proba(data16)
CTDC_EGAAC_CB_predict_proba = CB_CTDC_EGAAC.predict_proba(data17)

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
# Final_predict,Final_predict_proba=Final_predictor1(AAC_CB_predict_proba,CTDC_CB_predict_proba,EAAC_RF_predict_proba,EGAAC_CB_predict_proba,ESM_CB_predict_proba)
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
#
# Final_predict,Final_predict_proba=Final_predictor2(CTDC_CB_predict_proba,EAAC_RF_predict_proba,EGAAC_CB_predict_proba,ESM_CB_predict_proba)
#
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

# 3个
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
# Final_predict,Final_predict_proba=Final_predictor3(AAC_EAAC_CB_predict_proba,CTDC_EGAAC_CB_predict_proba,ESM_CB_predict_proba)
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
Final_predict,Final_predict_proba=Final_predictor4(CTDC_CB_predict_proba,ESM_CB_predict_proba)

#评价指标
# 计算SN和SP
tn, fp, fn, tp = confusion_matrix(target1, Final_predict).ravel()
SN = tp / (tp + fn)
SP = tn / (tn + fp)
fpr1, tpr1, thread1 = roc_curve(target1, Final_predict_proba)
roc_auc1 = auc(fpr1, tpr1)
print("ACC:{:.4f}".format(accuracy_score(target1, Final_predict)))
print("AUC:{:.4f}".format(roc_auc1))
print("SN:{:.4f}".format(SN))
print("SP:{:.4f}".format(SP))
print("MCC:{:.4f}".format(matthews_corrcoef(target1, Final_predict)))