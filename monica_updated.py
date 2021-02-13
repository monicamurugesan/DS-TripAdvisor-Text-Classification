# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:45:45 2020

@author: Hp
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


data_model1 = pd.read_csv("G:/Anaconda/envs/testenvs/Project-P39/model.csv")

data_model1.head()


data_model1.describe()


data_model1.shape


print("Data Label Count\n\n", data_model1["Label"].value_counts())


from sklearn.feature_extraction.text import CountVectorizer


count_vect = CountVectorizer(max_features = 3000)
x = count_vect.fit_transform(data_model1['Review']).toarray()

x.shape

pickle.dump(count_vect, open('cv-moni.pkl', 'wb'))

from sklearn import model_selection, preprocessing, svm, metrics
from sklearn.metrics import recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
# !pip install imbalanced-learn
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler, SMOTE

os=SMOTE(random_state=60,sampling_strategy='all')
data_oversample,y_label= os.fit_sample(x,data_model1["Label"])

X_train, X_test, y_train, y_test = train_test_split(data_oversample,y_label,test_size=0.25)




X_oversample,y_oversample=os.fit_sample(X_train,y_train)
# X_train_oversample, y_train_oversample= os.fit_sample(X_oversample,y_oversample)
# X_train_ovs,X_test_ovs,y_train_ovs,y_test_ovs=train_test_split(X_train_oversample,y_train_oversample,test_size=0.25)
# X_train_ovs.shape
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# sc.fit(X_train_oversample)
# X_t=sc.transform(X_train_oversample)
# X_te=sc.transform(X_test)
rf1 = RandomForestClassifier(n_estimators = 200, 
                            random_state = 60,
                            n_jobs = -1,
                            max_features = 'auto')
rf1.fit(X_train, y_train)
rf1_pred = rf1.predict(X_test)
result=rf1.score(X_test,y_test)
# y_train_ovs.shape
# y_test_ovs.shape
# X_test_ovs.shape
rf1_pred.shape
accuracy=result*100
#p1=rf1.transform(X_test,y_test)
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
rf1_enc = OneHotEncoder()
rf1_lm = LogisticRegression(max_iter=1000)

rf1_enc.fit(rf1.apply(X_train))
rf1_lm.fit(rf1_enc.transform(rf1.apply(X_train)), y_train)
print("Accuracy of Random Forest Classifier:\t", accuracy_score(y_test, rf1_pred))


print("Classification Report:\n\n",classification_report(y_test, rf1_pred))


print("Confusion Matrix:\n\n", confusion_matrix(y_test, rf1_pred))
rf1.estimators_

fimp=pd.Series(rf1.feature_importances_).sort_values(ascending=True)

print(fimp)
import seaborn as sns
sns.barplot(x=round(fimp,2),y=fimp)
plt.xlabel("Feature importance")
plt.show()
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
dt=tree.DecisionTreeClassifier(random_state = 0)
dt.fit(X_train,y_train)
dt_pred=dt.predict(X_test)

print("Accuracy of Decision Tree Classifier:\t", accuracy_score(y_test, dt_pred))

print("Classification Report:\n\n",classification_report(y_test, dt_pred))

print("Confusion Matrix:\n\n", confusion_matrix(y_test, dt_pred))


import graphviz 
dot_data1 = tree.export_graphviz(dt, out_file=None,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data1)  
graph.render(dot_data1) 


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB(alpha=1,fit_prior=True,class_prior=None)
nb.fit(X_train,y_train)
nb_pred=nb.predict(X_test)

print("Accuracy of Multinomial NB:\t", accuracy_score(y_test, nb_pred))

print("Classification Report:\n\n",classification_report(y_test, nb_pred))

print("Confusion Matrix:\n\n", confusion_matrix(y_test, nb_pred))


from sklearn import svm
s1=svm.LinearSVC(multi_class='ovr')
s1.fit(X_train,y_train)
s1_pred=s1.predict(X_test)


print("Accuracy of Support Vector Machine:\t", accuracy_score(y_test, s1_pred))

print("Classification Report:\n\n",classification_report(y_test, s1_pred))

print("Confusion Matrix:\n\n", confusion_matrix(y_test, s1_pred))


from sklearn.neighbors import KNeighborsClassifier
k1=KNeighborsClassifier(n_neighbors=4)
k1.fit(X_train,y_train)
k1_pred=k1.predict(X_test)


print("Accuracy of K-Nearest Neighbor:\t", accuracy_score(y_test, k1_pred))

print("Classification Report:\n\n",classification_report(y_test, k1_pred))

print("Confusion Matrix:\n\n", confusion_matrix(y_test, k1_pred))


###OneVsRest using AdaBoost
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import  AdaBoostClassifier

adao1=OneVsRestClassifier(AdaBoostClassifier())
adao1.fit(X_train,y_train)
adao1_pred=adao1.predict(X_test)

print("Accuracy of Adaptive Boosting:\t", accuracy_score(y_test, adao1_pred))

print("Classification Report:\n\n",classification_report(y_test, adao1_pred))

print("Confusion Matrix:\n\n", confusion_matrix(y_test, adao1_pred))

import xgboost as xg


ov1=OneVsRestClassifier(xg.XGBClassifier())
ov1.fit(X_train,y_train)
ov1_pred=ov1.predict(X_test)

print("Accuracy of XGBoosting:\t", accuracy_score(y_test, ov1_pred))

print("Classification Report:\n\n",classification_report(y_test, ov1_pred))

print("Confusion Matrix:\n\n", confusion_matrix(y_test, ov1_pred))



xgb = xgb.XGBClassifier(learning_rate = 0.01,
                       colsample_bytree = 0.8,
                       subsample = 0.8,
                       objective = 'multi:softmax', 
                       n_estimators = 100, 
                       reg_alpha = 0.3,
                       max_depth = 4, 
                       gamma = 1,
                       num_class = 3)


xgb.fit(X_train,y_train)


xgb_pred = xgb.predict(X_test)


# In[48]:


print("Accuracy of XGB =", accuracy_score(y_test,xgb_pred),"\n")
print("Classification of XGB\n\n",classification_report(y_test,xgb_pred),"\n")
print("Confusion matrix of XGB\n\n\n",confusion_matrix(y_test,xgb_pred))


from sklearn.linear_model import LogisticRegression


# In[39]:


LR = LogisticRegression(solver = 'liblinear',
                       multi_class = 'ovr',
                       max_iter = 1000,
                       random_state = 42,
                       penalty ="l2")
LR.fit(X_train,y_train)


# In[40]:


LR_pred = LR.predict(X_test)


# In[41]:


print("Accuracy of Logistic Regression\n\n", accuracy_score(y_test,LR_pred))


# In[42]:


print("Classification of Logistic Regression\n\n",classification_report(y_test,LR_pred))


# In[43]:


print("Confusion matrix of Logistic Regression\n\n\n",confusion_matrix(y_test,LR_pred))




from sklearn.linear_model import SGDClassifier


# In[33]:


sgd = SGDClassifier(alpha = 0.00047, random_state = 50)


# In[34]:


sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)


# In[35]:



print("Accuracy of SGD:\n\n", accuracy_score(y_test,sgd_pred))


# In[36]:


print("classification report of SGD\n\n", classification_report(y_test,sgd_pred))


# In[37]:


print("Confusion matrix sgd:\n\n", confusion_matrix(y_test,sgd_pred))


from sklearn.linear_model import Perceptron


# In[66]:


per = Perceptron(penalty = 'l1',alpha = 0.0005)
per.fit(X_train, y_train)


# In[67]:


per_pred = per.predict(X_test)


# In[68]:


print("Accuracy of  Perceptron=", accuracy_score(y_test,per_pred),"\n")
print("Classification of   Perceptron\n\n",classification_report(y_test,per_pred),"\n")
print("Confusion matrix of  Perceptron\n\n\n",confusion_matrix(y_test,per_pred))


# In[69]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[70]:


pac = PassiveAggressiveClassifier(class_weight = 'balanced', C = 0.4)
pac.fit(X_train, y_train)
pac_pred = pac.predict(X_test)


# In[71]:


print("Accuracy of  PassiveAggressiveClassifier=", accuracy_score(y_test,pac_pred),"\n")
print("Classification of   PassiveAggressiveClassifier\n\n",classification_report(y_test,pac_pred),"\n")
print("Confusion matrix of  PassiveAggressiveClassifier\n\n\n",confusion_matrix(y_test,pac_pred))


# In[72]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[73]:


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)


# In[74]:


lda_pred = lda.predict(X_test)


# In[75]:


print("Accuracy of  LinearDiscriminantAnalysi=", accuracy_score(y_test,lda_pred),"\n")
print("Classification of   LinearDiscriminantAnalysi\n\n",classification_report(y_test,lda_pred),"\n")
print("Confusion matrix of  LinearDiscriminantAnalysi\n\n\n",confusion_matrix(y_test,lda_pred))


# # RIDGE 

# In[76]:


from sklearn.linear_model import RidgeClassifier


# In[77]:


rc = RidgeClassifier()
rc.fit(X_train, y_train)


# In[78]:


rc_pred = rc.predict(X_test)


# In[79]:


print("Accuracy of  ridge=", accuracy_score(y_test,rc_pred),"\n")
print("Classification of   ridge\n\n",classification_report(y_test,rc_pred),"\n")
print("Confusion matrix of  ridge\n\n\n",confusion_matrix(y_test,rc_pred))



print("Accuracy of Random forest classifier=", accuracy_score(y_test,rf1_pred),'\n')
print("Accuracy of Multinomial Naive Bayes=", accuracy_score(y_test,nb_pred),'\n')
print("Accuracy of SGD=", accuracy_score(y_test,sgd_pred),'\n')
print("Accuracy of Logistic Regression=", accuracy_score(y_test,LR_pred),'\n')
print("Accuracy of svm=", accuracy_score(y_test,s1_pred),"\n")
print("Accuracy of ada =", accuracy_score(y_test,adao1_pred),"\n")
print("Accuracy of XGB =", accuracy_score(y_test,xgb_pred),"\n")
print("Accuracy of OneVsRestClassifier using xgb=", accuracy_score(y_test,ov1_pred),"\n")
print("Accuracy of knn=", accuracy_score(y_test,k1_pred),"\n")
print("Accuracy of Perceptron=", accuracy_score(y_test,per_pred),"\n")
print("Accuracy of PassiveAggressiveClassifier=", accuracy_score(y_test,pac_pred),"\n")
print("Accuracy of LinearDiscriminantAnalysi=", accuracy_score(y_test,lda_pred),"\n")
print("Accuracy of ridge=", accuracy_score(y_test,rc_pred),"\n")

pickle.dump(nb, open('monica1.pkl', 'wb'))
