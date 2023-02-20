#!/usr/bin/env python
# coding: utf-8

# In[24]:


# IMPORTING REQUIRED LIBARIES:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set()


# In[25]:


data = pd.read_csv("C:/Users/gtspy02/Desktop/phishing url/M1/Dataset/phishing_site_urls.csv")


# In[26]:


data.head(5)


# In[27]:


data.shape


# In[28]:


data.info()


# In[29]:


data.isnull().sum()


# In[30]:


data.drop_duplicates(subset="URL",keep=False, inplace=True)


# In[31]:


data.shape


# In[32]:


data["Label"].value_counts()


# In[33]:


sns.histplot(data=data["Label"])


# In[34]:


data_class_bad = data[data["Label"]=="bad"]
data_class_good = data[data["Label"]=="good"]


# In[35]:


data_class_bad.shape


# In[36]:


data_class_good.shape


# In[37]:


data_sample_bad = data_class_bad.sample(2500)


# In[38]:


data_sample_good = data_class_good.sample(2500)


# In[39]:


data_1 = pd.concat([data_sample_bad,data_sample_good],axis=0)


# In[40]:


data_1.shape


# In[41]:


data_1.info()


# In[42]:


data_1["Label"].value_counts()


# In[43]:


data_1.to_csv("C:/Users/gtspy02/Desktop/phishing url/M2/Preprocessed dataset/preprocessed_dataset.csv",index=False)

