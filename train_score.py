# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import StandardScaler
from collections import Counter
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time
#from sklearn.feature_selection import RFE


start = time.clock()

data=pd.read_csv('D:/ecg12/feature_train.csv',na_values='\\N',encoding='gbk')
label=pd.read_csv('D:/ecg12/REFERENCE.csv',na_values='\\N',encoding='gbk')


colnames=list(data.columns.values)
feature_drop=['Recording','PR_min_V5','PR_min_V6','QRS_duration_std_V3','R_amp_std_V6',
            'QRS_duration_std_V2','T_amp_ave_V4','PR_min_V4','T_amp_ave_aVL','T_amp_ave_V5',
            'T_amp_ave_V3','QRS_duration_std_III','PP_max_V3','R_amp_max_aVF','S_amp_min_aVF']

data_num=data.drop(feature_drop,axis=1)
x_train,x_test,y_train,y_test=train_test_split(data_num,label,random_state=33,test_size=0.3)
x_train=x_train.fillna(x_train.mean())

y_train=y_train['First_label']
x_train=x_train.as_matrix().astype(float)
y_train=y_train.as_matrix().astype(float)
x_test=x_test.as_matrix().astype(float)
reference=dict()
for i in range(len(y_test)):
    reference.setdefault(y_test.iloc[i,0],[]).append([y_test.iloc[i,1], y_test.iloc[i,2], y_test.iloc[i,3]])
    

def confusion_matrix_score(test_pred_label):
    cf = np.zeros((9, 9), dtype=np.float)
    i=0
    for key in reference.keys():
        value = []
        predict=np.int(test_pred_label[i])
        for item in reference[key][0]:
            if item == ' ':
                item = 0
            value.append(item)   
        if predict in value:
            cf[predict-1][predict-1] += 1
        else:
            cf[value[0]-1][predict-1] += 1
        i=i+1
    return cf

def ecg_score(cf):
    Normal=2*(cf[0,0])/(sum(cf[0,:])+sum(cf[:,0]))
    AF=2*(cf[1,1])/(sum(cf[1,:])+sum(cf[:,1]))
    I_AVF=2*(cf[2,2])/(sum(cf[2,:])+sum(cf[:,2]))
    LBBB=2*(cf[3,3])/(sum(cf[3,:])+sum(cf[:,3]))
    RBBB=2*(cf[4,4])/(sum(cf[4,:])+sum(cf[:,4]))
    PAC=2*(cf[5,5])/(sum(cf[5,:])+sum(cf[:,5]))
    PVC=2*(cf[6,6])/(sum(cf[6,:])+sum(cf[:,6]))
    STD=2*(cf[7,7])/(sum(cf[7,:])+sum(cf[:,7]))
    STE=2*(cf[8,8])/(sum(cf[8,:])+sum(cf[:,8]))
    F1=(Normal+AF+I_AVF+LBBB+RBBB+PAC+PVC+STD+STE)/9
    return Normal,AF,I_AVF, LBBB, RBBB, PAC,PVC,STD,STE,F1




####xgboost模型处理结果

#####做一下特征选择
#rfe=RFE(estimator=model1, n_features_to_select=84,step=1,verbose=1)
#rfe.fit(x_train,y_train)
#rfe.n_features_
#plt.figure()
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")
#plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
#plt.show()
#col = []
#for i in range(len(colnames)):
#    if rfe.support_[i]:
#        col.append(colnames[i])


#print('processing xgboost.............')
#depth=[2,3,4,5]
#numbers=[200,250,300,350,400]
#for i in depth:
#    for j in numbers:
model1=XGBClassifier(learning_rate=0.1,max_depth=4,scale_pos_weight=1,n_estimators=385)
model1.fit(x_train,y_train)
feature=model1.feature_importances_
idxsorted=np.argsort(-feature)
xgb_feature=[colnames[i] for i in idxsorted]
xgb_feature_score=[feature[i] for i in idxsorted]
xgb_feature_final=pd.DataFrame(xgb_feature_score,index=xgb_feature,columns=['feature_score'])
        
train_pred=model1.predict_proba(x_train)
test_pred=model1.predict_proba(x_test)
train_pred_label=model1.predict(x_train)
test_pred_label=model1.predict(x_test)
                        
xgb_cf=confusion_matrix_score(test_pred_label)
xgb_acc=Counter(test_pred_label==y_test['First_label'])[1]/len(y_test['First_label'])
xgb_acc_train=Counter(train_pred_label==y_train)[1]/len(y_train)
Normal,AF,I_AVF, LBBB, RBBB, PAC,PVC,STD,STE,F1=ecg_score(xgb_cf)
print('depth为',4,'n_estimators为',385,'f1为',F1)

#############lightgbm模型处理结果
#rfe=RFE(estimator=model2,n_features_to_select=84,step=1,verbose=1)
#rfe.fit(x_train,y_train)
#col = []
#for i in range(len(colnames)-1):
#    if rfe.support_[i]:
#        col.append(colnames[i])
#        print(colnames[i])
#
#feature_select=[i for i in range(len(rfe.ranking_)) if rfe.ranking_[i]!=1 ]
#x_train=x_train[:,feature_select]
#x_test=x_test[:,feature_select]

print('processing lightgbm.............')
model2=LGBMClassifier(learning_rate=0.1,max_depth=3,num_leaves=15,n_estimators=300)
model2.fit(x_train,y_train)
feature=model2.feature_importances_
idxsorted=np.argsort(-feature)
lgb_feature=[colnames[i] for i in idxsorted]
lgb_feature_score=[feature[i] for i in idxsorted]
lgb_feature_final=pd.DataFrame(lgb_feature_score,index=lgb_feature,columns=['feature_score'])

train_pred=model2.predict_proba(x_train)
test_pred=model2.predict_proba(x_test)
train_pred_label=model2.predict(x_train)
test_pred_label=model2.predict(x_test)

lgb_cf=confusion_matrix_score(test_pred_label)
lgb_acc=Counter(test_pred_label==y_test['First_label'])[1]/len(y_test['First_label'])
lgb_acc_train=Counter(train_pred_label==y_train)[1]/len(y_train)
Normal,AF,I_AVF, LBBB, RBBB, PAC,PVC,STD,STE,F1=ecg_score(lgb_cf)
print('acc为',lgb_acc,'f1为',F1)

#lgb_feature_final.to_csv('D:/ecg12/feture_score_train.csv')
elapsed = (time.clock() - start)
print("Time used:",elapsed)

