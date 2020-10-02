# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 21:31:35 2020

@author: Abhinash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r"C:\Users\Abhinash\Downloads\UNSW_NB15_training-set.csv")
print(data.head())

#Explore Data Description

data.info()

#check % of missing value
per=data.isnull().sum()/len(data)
per

#no missing value in data
data.isnull().sum()

data.describe()

#lets check what type of attack(variety)
data['attack_cat'].value_counts()

#lets ovserve the success and unsucessful attack
data["label"].value_counts()

#object type label encode
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['attack_cat']=le.fit_transform(data['attack_cat'])
data['proto']=le.fit_transform(data['proto'])
data['service']=le.fit_transform(data['service'])
data['state']=le.fit_transform(data['state'])
 
data=data.drop(['ct_ftp_cmd','is_ftp_login',
                'tcprtt','synack','ackdat','id'],axis=1)
 

#lets find each sucessful and unsucessful in each Caetgory
#EDA:Explotary data analysis

#feature selections
#seperate dependent and independent variable
x=data.drop(["label"],axis=1)
y=data["label"]


import statsmodels.api as sm
x_con=sm.add_constant(x)
logit=sm.Logit(y,x_con).fit()
logit.summary()

x1=x_con.drop(["is_sm_ips_ports"],axis=1)
logit=sm.Logit(y,x1).fit()
logit.summary()

x2=x1.drop(["spkts"],axis=1)
logit=sm.Logit(y,x2).fit()
logit.summary()

x3=x2.drop(["sbytes"],axis=1)
logit=sm.Logit(y,x3).fit()
logit.summary()


x4=x3.drop(["sloss"],axis=1)
logit=sm.Logit(y,x4).fit()
logit.summary()

x5=x4.drop(["response_body_len"],axis=1)
logit=sm.Logit(y,x5).fit()
logit.summary()

x6=x5.drop(["sjit"],axis=1)
logit=sm.Logit(y,x6).fit()
logit.summary()


x7=x6.drop(["stcpb"],axis=1)
logit=sm.Logit(y,x7).fit()
logit.summary()

x8=x7.drop(["dur"],axis=1)
logit=sm.Logit(y,x8).fit()
logit.summary()

x9=x8.drop(["trans_depth"],axis=1)
logit=sm.Logit(y,x9).fit()
logit.summary()


x10=x9.drop(["dtcpb"],axis=1)
logit=sm.Logit(y,x10).fit()
logit.summary()


x11=x10.drop(["sload"],axis=1)
logit=sm.Logit(y,x11).fit()
logit.summary()


x12=x11.drop(["ct_src_dport_ltm"],axis=1)
logit=sm.Logit(y,x12).fit()
logit.summary()

#VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor 
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = x12.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(x12.values, i) 
                          for i in range(len(x12.columns))] 
  
print(vif_data)

#final data
vif_df=vif_data[vif_data['VIF']<10]
vif_df

col=[]
for i in vif_df["feature"]:
    col.append(i)
col

#only use these col for ml-model

X=data.loc[0:,col]
X.columns
Y=data["label"]

#model
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC


nb=GaussianNB()
knn=KNeighborsClassifier()
lr=LogisticRegression()
#svm=SVC()

#model fit
nb.fit(X,Y)
knn.fit(X,Y)
lr.fit(X,Y)
#svm.fit(X,Y)
#testing the model load test file

test=pd.read_csv(r"C:\Users\Abhinash\Downloads\UNSW_NB15_testing-set.csv")
test.columns


y_test=test["label"]
test=test.loc[0:,col]
test.info()

#object type label encode
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
test['attack_cat']=le.fit_transform(test['attack_cat'])
test['proto']=le.fit_transform(test['proto'])
test['service']=le.fit_transform(test['service'])
test['state']=le.fit_transform(test['state'])
test.info()

#prediction
nb_pred=nb.predict(test)
knn_pred=knn.predict(test)
lr_pred=lr.predict(test)
#svm_pred=svm.predict(test)

from sklearn.metrics import accuracy_score
print("Accuracy of model is:",accuracy_score(y_test,nb_pred))
print("Accuracy of model is:",accuracy_score(y_test,knn_pred))
print("Accuracy of model is:",accuracy_score(y_test,lr_pred))
#print("Accuracy of model is:",accuracy_score(y_test,svm_pred))

#result 
mdl=['LogisticRegression','KNN','NaiveBayes']
acc=[accuracy_score(y_test,lr_pred),
     accuracy_score(y_test,knn_pred),
     accuracy_score(y_test,nb_pred)]

res=pd.DataFrame({"Model":mdl,
                  "Accuracy":acc})
print(res)

plt.bar(res['Model'],res['Accuracy'],width=0.5)
plt.plot(res['Model'],res['Accuracy'])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()

