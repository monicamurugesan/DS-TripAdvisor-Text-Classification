#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("model.csv")


# In[3]:


print("Data Head\n",data.head())


# In[4]:


print("Data Describe\n", data.describe())


# In[5]:


print("Data Shape\n",data.shape)


# In[6]:


print("Data Label Count\n\n", data["Label"].value_counts())


# In[7]:


plt.hist(data.Label)
plt.title("LABEL")
plt.show()


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
import pickle

# In[9]:


count_vect = CountVectorizer(max_features = 5000)
x = count_vect.fit_transform(data['Review']).toarray()


pickle.dump(count_vect, open('cv-moni2.pkl', 'wb'))
# In[10]:


x.shape


# In[11]:


#pip install --user -U imblearn


# In[12]:


from imblearn.over_sampling import SMOTE


# In[13]:


over_sample = SMOTE(random_state = 100, sampling_strategy = "all")


# In[14]:


data_oversample,y_label = over_sample.fit_sample(x,data['Label'])


# In[15]:


y_label.value_counts()


# In[16]:


data_oversample.shape


# ###### There is a data imbalance 

# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x_train,x_test,y_train,y_test = train_test_split(data_oversample,y_label, test_size = 0.3, random_state = 42)


# In[19]:


print(x_train.shape,y_train.shape)
print(x_test.shape, y_test.shape)


# In[20]:


colors = ['lime'] 
  
plt.hist(y_train, 
         density = True,  
         histtype ='barstacked', 
         color = colors)  
  
plt.title('balanced Data\n\n', 
          fontweight ="bold") 
  
plt.show() 


# # Ensemble 
# 
# 
# #### Random_Forest_bagging

# In[21]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,classification_report, confusion_matrix


# In[22]:


RF = RandomForestClassifier(n_estimators = 120,
                           random_state = 50,
                           n_jobs = -1,
                           max_features = 'auto')
RF.fit(x_train,y_train)


# In[23]:


rf_pred = RF.predict(x_test)


# In[24]:


print("Accuracy of Random forest classifier:\n\n", accuracy_score(y_test,rf_pred))


# In[25]:


print("Classification Report:\n\n", classification_report(y_test,rf_pred))


# In[26]:


print("Confusion Matrix \n\n", confusion_matrix(y_test,rf_pred))


# # MultiNomial Naive Bayes

# In[27]:


from sklearn.naive_bayes import MultinomialNB


# In[28]:


mn = MultinomialNB()
mn.fit(x_train, y_train)


# In[29]:


mn_pred = mn.predict(x_test)


# In[30]:


print("Accuracy of Multinomial Naive Bayes\n\n\n", accuracy_score(y_test,mn_pred))


# In[31]:


print("Classification Report \n\n\n", classification_report(y_test,mn_pred))


# In[32]:


print("confusion matrix:\n\n\n", confusion_matrix(y_test,mn_pred))


# # Stochastic Gradient Descent 
# 
# 
# ##### Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.

# In[37]:


from sklearn.linear_model import SGDClassifier


# In[38]:


sgd = SGDClassifier(alpha = 0.00047, random_state = 50)


# In[39]:


sgd.fit(x_train, y_train)
sgd_pred = sgd.predict(x_test)


# In[40]:



print("Accuracy of SGD:\n\n", accuracy_score(y_test,sgd_pred))


# In[41]:


print("classification report of SGD\n\n", classification_report(y_test,sgd_pred))


# In[42]:


print("Confusion matrix sgd:\n\n", confusion_matrix(y_test,sgd_pred))


# # Logistic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


LR = LogisticRegression(solver = 'liblinear',
                       multi_class = 'ovr',
                       max_iter = 1000,
                       random_state = 42,
                       penalty ="l2")
LR.fit(x_train,y_train)


# In[45]:


LR_pred = LR.predict(x_test)


# In[46]:


print("Accuracy of Logistic Regression\n\n", accuracy_score(y_test,LR_pred))


# In[47]:


print("Classification of Logistic Regression\n\n",classification_report(y_test,LR_pred))


# In[48]:


print("Confusion matrix of Logistic Regression\n\n\n",confusion_matrix(y_test,LR_pred))


# # XGBoost

# In[50]:


#pip install --user -U xgboost


# In[51]:




import xgboost as xg


# In[52]:


xgb = xg.XGBClassifier(learning_rate = 0.01,
                       colsample_bytree = 0.8,
                       subsample = 0.8,
                       objective = 'multi:softmax', 
                       n_estimators = 100, 
                       reg_alpha = 0.3,
                       max_depth = 4, 
                       gamma = 1,
                       num_class = 3)


# In[53]:


xgb.fit(x_train , y_train)


# In[54]:


xgb_pred = xgb.predict(x_test)


# In[57]:


print("Accuracy of XGB =", accuracy_score(y_test,xgb_pred),"\n")
print("Classification of XGB\n\n",classification_report(y_test,xgb_pred),"\n")
print("Confusion matrix of XGB\n\n\n",confusion_matrix(y_test,xgb_pred))


# In[58]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import  AdaBoostClassifier


# In[59]:


ada_model = OneVsRestClassifier(AdaBoostClassifier())


# In[60]:


ada_model.fit(x_train,y_train)


# In[61]:


ada_pred = ada_model.predict(x_test)


# In[64]:


print("Accuracy of ada =", accuracy_score(y_test,ada_pred),"\n")
print("Classification of ada\n\n",classification_report(y_test,ada_pred),"\n")
print("Confusion matrix of ada\n\n\n",confusion_matrix(y_test,ada_pred))


# In[64]:


ovr_model = OneVsRestClassifier(xg.XGBClassifier())
ovr_model.fit(x_train,y_train)


# In[65]:


ovr_model_pred = ovr_model.predict(x_test)


# In[66]:


print("Accuracy of OneVsRestClassifier using xgb=", accuracy_score(y_test,ovr_model_pred),"\n")
print("Classification of  OneVsRestClassifier using xgb\n\n",classification_report(y_test,ovr_model_pred),"\n")
print("Confusion matrix of  OneVsRestClassifier using xgb\n\n\n",confusion_matrix(y_test,ovr_model_pred))


# # suport vector classifier

# In[67]:


from sklearn import svm


# In[68]:


svm = svm.LinearSVC(multi_class = 'ovr')
svm.fit(x_train,y_train)


# In[69]:


svm_pred = svm.predict(x_test)


# In[70]:


print("Accuracy of svm=", accuracy_score(y_test,svm_pred),"\n")
print("Classification of  svm\n\n",classification_report(y_test,svm_pred),"\n")
print("Confusion matrix of svm\n\n\n",confusion_matrix(y_test,svm_pred))


# # KNN classification

# In[71]:


from sklearn.neighbors import KNeighborsClassifier


# In[72]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)


# In[73]:


knn_pred = knn.predict(x_test)


# In[76]:


print("Accuracy of knn=", accuracy_score(y_test,knn_pred),"\n")
print("Classification of  knn\n\n",classification_report(y_test,knn_pred),"\n")
print("Confusion matrix of knn\n\n\n",confusion_matrix(y_test,knn_pred))


# In[77]:


from sklearn.linear_model import Perceptron


# In[78]:


per = Perceptron(penalty = 'l1',alpha = 0.0005)
per.fit(x_train,y_train)


# In[79]:


per_pred = per.predict(x_test)


# In[80]:


print("Accuracy of  Perceptron=", accuracy_score(y_test,per_pred),"\n")
print("Classification of   Perceptron\n\n",classification_report(y_test,per_pred),"\n")
print("Confusion matrix of  Perceptron\n\n\n",confusion_matrix(y_test,per_pred))


# In[80]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[81]:


pac = PassiveAggressiveClassifier(class_weight = 'balanced',C = 0.4)
pac.fit(x_train,y_train)
pac_pred = pac.predict(x_test)


# In[82]:


print("Accuracy of  PassiveAggressiveClassifier=", accuracy_score(y_test,pac_pred),"\n")
print("Classification of   PassiveAggressiveClassifier\n\n",classification_report(y_test,pac_pred),"\n")
print("Confusion matrix of  PassiveAggressiveClassifier\n\n\n",confusion_matrix(y_test,pac_pred))


# In[83]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[84]:


lda = LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)


# In[85]:


lda_pred = lda.predict(x_test)


# In[86]:


print("Accuracy of  LinearDiscriminantAnalysi=", accuracy_score(y_test,lda_pred),"\n")
print("Classification of   LinearDiscriminantAnalysi\n\n",classification_report(y_test,lda_pred),"\n")
print("Confusion matrix of  LinearDiscriminantAnalysi\n\n\n",confusion_matrix(y_test,lda_pred))


# # RIDGE 

# In[87]:


from sklearn.linear_model import RidgeClassifier


# In[88]:


rc = RidgeClassifier()
rc.fit(x_train,y_train)


# In[89]:


rc_pred = rc.predict(x_test)


# In[90]:


print("Accuracy of  ridge=", accuracy_score(y_test,lda_pred),"\n")
print("Classification of   ridge\n\n",classification_report(y_test,lda_pred),"\n")
print("Confusion matrix of  ridge\n\n\n",confusion_matrix(y_test,lda_pred))


# In[91]:


print("Accuracy of Random forest classifier=", accuracy_score(y_test,rf_pred),'\n')
print("Accuracy of Multinomial Naive Bayes=", accuracy_score(y_test,mn_pred),'\n')
print("Accuracy of SGD=", accuracy_score(y_test,sgd_pred),'\n')
print("Accuracy of Logistic Regression=", accuracy_score(y_test,LR_pred),'\n')
print("Accuracy of svm=", accuracy_score(y_test,svm_pred),"\n")
print("Accuracy of ada =", accuracy_score(y_test,ada_pred),"\n")
print("Accuracy of XGB =", accuracy_score(y_test,xgb_pred),"\n")
print("Accuracy of OneVsRestClassifier using xgb=", accuracy_score(y_test,ovr_model_pred),"\n")
print("Accuracy of knn=", accuracy_score(y_test,knn_pred),"\n")
print("Accuracy of Perceptron=", accuracy_score(y_test,per_pred),"\n")
print("Accuracy of PassiveAggressiveClassifier=", accuracy_score(y_test,pac_pred),"\n")
print("Accuracy of LinearDiscriminantAnalysi=", accuracy_score(y_test,lda_pred),"\n")
print("Accuracy of ridge=", accuracy_score(y_test,lda_pred),"\n")


# # finally random forest classifier is the best model to deploy

# In[ ]:



import pickle
pickle.dump(RF, open('monica-r4.pkl', 'wb'))
