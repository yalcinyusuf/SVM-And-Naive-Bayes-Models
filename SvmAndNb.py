#!/usr/bin/env python
# coding: utf-8
# Muhammed Ali Al Houssını 18120212006
# Mustafa Kemal Gökçe  18120205034 
# Yusuf YALÇIN 18120205032 

# In[1]:


import warnings
warnings.filterwarnings('ignore')


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd 
import numpy as np

data1 = pd.read_csv('data_Mar_64.txt', header= None)
data2 = pd.read_csv('data_Sha_64.txt', header= None)
data3 = pd.read_csv('data_Tex_64.txt', header= None)



data1 = data1.sort_values(by=data1.columns[0]).iloc[:,:]
data2 = data2.sort_values(by=data2.columns[0]).iloc[:,1:]
data3 = data3.sort_values(by=data3.columns[0]).iloc[:,1:]

data2 = data2.reset_index(drop = True)
data3 = data3.reset_index(drop = True)

data = pd.concat([data1, data2, data3], axis=1, ignore_index=True)


X = data.iloc[:,1:].values
y = data.iloc[:,:1].values

y = LabelEncoder().fit_transform(y)


gaussianModel = GaussianNB(var_smoothing = 1e-9)
svmModel = SVC(kernel= 'poly')


def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (accuracy_score(y, yPred), 
            precision_score(y, yPred,average='macro',zero_division=0), 
            recall_score(y, yPred, average='macro',zero_division=0),
            f1_score(y, yPred, average='macro',zero_division=0),)

def my_scorer(estimator, x, y):
    acc, pre, rcl, f1 = getScores(estimator, x, y)
    print ("acc:",acc,"pre:", pre,"recl:", rcl, "f1:", f1)
    return acc

cv = KFold(n_splits=3, shuffle=True, random_state=1)
gaussianScores = cross_val_score(estimator=gaussianModel,X= X,y= y, scoring=my_scorer, cv=cv)
print(gaussianScores.argmax(), ". score gauss modelinde en buyuk accuracye sahip",'Accuray değeri:', gaussianScores[gaussianScores.argmax()])
print("-------------------------------------------------------------------------------------")

cv = KFold(n_splits=5, shuffle=True, random_state=1)
svmScores = cross_val_score(estimator=svmModel,X= X,y= y, scoring=my_scorer, cv=cv)
print(svmScores.argmax(), ". score svm modelinde en buyuk accuracye sahip", 'Accuray değeri:',svmScores[svmScores.argmax()])


# 

# In[9]:


data_z = data.sort_values(by=data.columns[0]).iloc[:,1:]


cols = list(data_z.columns)
#data[cols]

for col in cols:
    data[col] = (data[col] - data[col].mean())/data[col].std(ddof=0)

X = data.iloc[:,1:].values
y = data.iloc[:,:1].values

y = LabelEncoder().fit_transform(y)

gaussianModel = GaussianNB(var_smoothing = 1e-4)
svmModel = SVC(kernel= 'rbf')

cv = KFold(n_splits=5, shuffle=True, random_state=1)
gaussianScores = cross_val_score(estimator=gaussianModel,X= X,y= y, scoring=my_scorer, cv=cv)
print(gaussianScores.argmax(), ". score gauss modelinde en buyuk accuracye sahip",'Accuray değeri:', gaussianScores[gaussianScores.argmax()])
print("-------------------------------------------------------------------------------------")

cv = KFold(n_splits=5, shuffle=True, random_state=1)
svmScores = cross_val_score(estimator=svmModel,X= X,y= y, scoring=my_scorer, cv=cv)
print(svmScores.argmax(), ". score svm modelinde en buyuk accuracye sahip", 'Accuray değeri:',svmScores[svmScores.argmax()])


# In[ ]:




