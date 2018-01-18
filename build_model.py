
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sc
import math
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from PreProcess import *
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[2]:


#Load data
filePath = "train_potus_by_county.csv"
testDf = pd.read_csv(filePath)

#testDf.isnull().values.any()  #check for nulls
y=testDf["Winner"]
X=testDf.drop(labels=["Winner"],axis=1)

#Preprocess the data, normallize features and removes few features and adds few useful ones
X=PreProcess(X)

thres=int(len(X)*0.80)

X_train = X[:thres]
X_test = X[thres:]
y_train = y[:thres]
y_test = y[thres:]

Models={}


# In[3]:


#Support Vector Classifier
clf = SVC(kernel = 'rbf', C = 1)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

Models["SVC"]=(clf,accuracy_score(y_test,y_pred))


# In[4]:


#Random Forest Classifier
rfc_clf = RandomForestClassifier(max_depth=None, random_state=0)
rfc_clf.fit(X_train,y_train)
y_pred = rfc_clf.predict(X_test)

Models["RFC"]=(rfc_clf,accuracy_score(y_test,y_pred))
'''c=0
for i in X:
    if rfc_clf.feature_importances_[c]<0.05:
        print i,rfc_clf.feature_importances_[c]
    c+=1'''


# In[5]:


#Multi Layer Perceptron
clf_neural = MLPClassifier(solver='lbfgs',activation='relu', alpha=1e-2,hidden_layer_sizes=(3,2,7,9),max_iter=10, random_state=1)
clf_neural.fit(X_train,y_train)
y_pred=clf_neural.predict(X_test)

Models["MLP"]=(clf_neural,accuracy_score(y_test,y_pred))


# In[6]:


#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

Models["GNB"]=(gnb,accuracy_score(y_test,y_pred))


# In[8]:


#Save the model with max accuracy
perffile=open("performance.txt",'w')
perffile.write("Accuracy Scores:\n")
maxacc=0
for i in Models:
    perffile.write(i+"--"+str(Models[i][1])+"\n")
    if(Models[i][1]>maxacc):
        model=Models[i][0]
        maxacc=Models[i][1]
print model,maxacc

f=open("model.pkl",'w')
pickle.dump(model, f)

perffile.close()
f.close()

