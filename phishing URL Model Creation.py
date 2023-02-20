#!/usr/bin/env python
# coding: utf-8

# In[179]:


# IMPORTING REQUIRED LIBARIES:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import warnings
warnings.simplefilter("ignore")


# In[180]:


data=pd.read_csv("C:/Users/gtspy02/Desktop/phishing url/M2/Preprocessed dataset/preprocessed_dataset.csv")


# In[181]:


data.head(5)


# In[182]:


import nltk
import re

from nltk.tokenize import RegexpTokenizer


# In[183]:


tokenizer = RegexpTokenizer(r'[A-Za-z]+')


# In[184]:


data.URL[4]


# In[185]:


tokenizer.tokenize(data.URL[4]) # using the fourth row..


# In[186]:


data['text_tokenized'] = data.URL.map(lambda t: tokenizer.tokenize(t))


# In[187]:


data


# In[188]:


# Stemming To find the rootwords:
from nltk.stem.snowball import SnowballStemmer


# In[189]:


stemmer = SnowballStemmer("english")


# In[190]:


data['text_stemmed'] = data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])


# In[191]:


data['text_sent'] = data['text_stemmed'].map(lambda l: ' '.join(l))


# In[192]:


data


# In[193]:


data.shape


# In[194]:


from sklearn.feature_extraction.text import CountVectorizer


# In[195]:


cv = CountVectorizer()


# In[196]:


feature = cv.fit_transform(data.text_sent)


# In[197]:


feature


# In[198]:


feature = feature[:].toarray()


# In[199]:


feature


# In[200]:


#from sklearn.feature_extraction.text import TfidfVectorizer


# In[201]:


#vectorizer = TfidfVectorizer()
#vectors = vectorizer.fit_transform(data.text_sent)
#vectors.shape


# In[202]:


data["Label"] = data["Label"].map({"bad":1,"good":0})


# In[203]:


X=feature


# In[263]:


print(X.shape)
print(data.shape)


# In[205]:


Y=data["Label"]


# In[206]:


Y


# In[207]:


from sklearn.model_selection import train_test_split


# In[208]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=42)


# In[209]:


X_train


# In[210]:


Y_train


# In[211]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[261]:


X_train


# In[212]:


from sklearn.linear_model import LogisticRegression


# In[213]:


LR=LogisticRegression()


# In[214]:


LR.fit(X_train,Y_train)
LR.score(X_train,Y_train)


# In[215]:


from sklearn.naive_bayes import GaussianNB


# In[216]:


NB=GaussianNB()


# In[217]:


NB.fit(X_train,Y_train)
NB.score(X_train,Y_train)


# In[218]:


from sklearn.svm import SVC


# In[219]:


SVM=SVC(kernel='rbf')


# In[220]:


SVM.fit(X_train,Y_train)
SVM.score(X_train,Y_train)


# In[221]:


from sklearn.neighbors import KNeighborsClassifier


# In[222]:


KNN=KNeighborsClassifier(n_neighbors=1)
KNN.fit(X_train,Y_train)
KNN.score(X_train,Y_train)


# In[223]:


from sklearn.ensemble import RandomForestClassifier


# In[224]:


RF=RandomForestClassifier(n_estimators=250)
RF.fit(X_train,Y_train)
RF.score(X_train,Y_train)


# In[225]:


from sklearn.ensemble import AdaBoostClassifier


# In[226]:


AdB=AdaBoostClassifier(n_estimators=50)
AdB.fit(X_train,Y_train)
AdB.score(X_train,Y_train)


# In[227]:


from sklearn.ensemble import GradientBoostingClassifier


# In[228]:


XgB=GradientBoostingClassifier(n_estimators=100)
XgB.fit(X_train,Y_train)
XgB.score(X_train,Y_train)


# In[229]:


from lightgbm import LGBMClassifier


# In[230]:


Lgbm=LGBMClassifier(num_leaves=100)
Lgbm.fit(X_train,Y_train)
Lgbm.score(X_train,Y_train)


# In[231]:


# Test score for all Algorithms:
print("****Test_score*****")
print("LR:",LR.score(X_test,Y_test))
print("NB:",NB.score(X_test,Y_test))
print("SVM:",SVM.score(X_test,Y_test))
print("KNN:",KNN.score(X_test,Y_test))
print("RF:",RF.score(X_test,Y_test))
print("AdB:",AdB.score(X_test,Y_test))
print("XgB:",XgB.score(X_test,Y_test))
print("Lgbm:",Lgbm.score(X_test,Y_test))


# In[232]:


y_LR_pred=LR.predict(X_test)
y_NB_pred=NB.predict(X_test)
y_SVM_pred=SVM.predict(X_test)
y_KNN_pred=KNN.predict(X_test)
y_RF_pred=RF.predict(X_test)
y_AdB_pred=AdB.predict(X_test)
y_XgB_pred=XgB.predict(X_test)
y_Lgbm_pred=Lgbm.predict(X_test)


# In[233]:


from sklearn import metrics
from sklearn.metrics import precision_score,recall_score,accuracy_score


# In[234]:


print("precision_score for LogisticRegression:",metrics.precision_score(Y_test,y_LR_pred))
print("Precision_score for Navie_bayes:",metrics.precision_score(Y_test,y_NB_pred))
print("Precision_score for Support vector Machine:",metrics.precision_score(Y_test,y_SVM_pred))
print("Precision_score for Knearestneighbors:",metrics.precision_score(Y_test,y_KNN_pred))
print("Precision_score for RandomForestclassifier:",metrics.precision_score(Y_test,y_RF_pred))
print("Precision_score for Adaboost classifier:",metrics.precision_score(Y_test,y_AdB_pred))
print("Precision_score for GradientBoostingclassifier:",metrics.precision_score(Y_test,y_XgB_pred))
print("Precision_score for LightGBM:",metrics.precision_score(Y_test,y_Lgbm_pred))


# In[235]:


print("RECALL_score for LogisticRegression:",metrics.recall_score(Y_test,y_LR_pred))
print("RECALL_score for Navie_bayes:",metrics.recall_score(Y_test,y_NB_pred))
print("RECALL_score for Support vector Machine:",metrics.recall_score(Y_test,y_SVM_pred))
print("RECALL_score for Knearestneighbors:",metrics.recall_score(Y_test,y_KNN_pred))
print("RECALL_score for RandomForestclassifier:",metrics.recall_score(Y_test,y_RF_pred))
print("RECALL_score for Adaboost classifier:",metrics.recall_score(Y_test,y_AdB_pred))
print("RECALL_score for GradientBoostingclassifier:",metrics.recall_score(Y_test,y_XgB_pred))
print("RECALL_score for LightGBM:",metrics.recall_score(Y_test,y_Lgbm_pred))


# In[236]:


print("ACCURACY_score for LogisticRegression:",metrics.accuracy_score(Y_test,y_LR_pred))
print("ACCURACY_score for Navie_bayes:",metrics.accuracy_score(Y_test,y_NB_pred))
print("ACCURACY_score for Support vector Machine:",metrics.accuracy_score(Y_test,y_SVM_pred))
print("ACCURACY_score for Knearestneighbors:",metrics.accuracy_score(Y_test,y_KNN_pred))
print("ACCURACY_score for RandomForestclassifier:",metrics.accuracy_score(Y_test,y_RF_pred))
print("ACCUARCY_score for Adaboost classifier:",metrics.accuracy_score(Y_test,y_AdB_pred))
print("ACCURACY_score for GradientBoostingclassifier:",metrics.accuracy_score(Y_test,y_XgB_pred))
print("ACCUARCY_score for LightGBM:",metrics.accuracy_score(Y_test,y_Lgbm_pred))


# In[237]:


from sklearn.pipeline import make_pipeline


# In[238]:


pipeline= make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'),LR)


# In[239]:


trainX, testX, trainY, testY = train_test_split(data.URL, data.Label)


# In[240]:


pipeline.fit(trainX,trainY)


# In[249]:


pipeline.score(testX,testY) 


# In[266]:


import pickle
model_predict=pickle.dump(pipeline,open('C:/Users/gtspy02/Desktop/phishing/M2/phishing.pkl','wb'))


# In[267]:


print(model_predict)


# In[257]:


loaded_model = pickle.load(open('C:/Users/gtspy02/Desktop/phishing/M2/phishing.pkl', 'rb'))


# In[268]:


print(loaded_model)


# In[258]:


result = loaded_model.score(testX,testY)
print(result)


# In[259]:


test=[data["URL"][4995]]
result3 = loaded_model.predict(test)
result3


# In[260]:


prediction=['www.atpa.cl/foros/install/wp-content/cimb/index/','service.confirm.paypal.cmd.cgi-bin.2466sd4f3e6... ','retailhellunderground.com/','restorevisioncenters.com/html/technology.html',"https://www.google.com/search?client=firefox-b-d&q=amazon+prime+membership","https://www.google.com/search?client=firefox-b-d&q=amazon+prime"]
result=loaded_model.predict(prediction)
result

