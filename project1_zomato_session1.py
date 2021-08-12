#!/usr/bin/env python
# coding: utf-8

# In[1]:


## read this file-manipulated file
## real world scenario - raw data [csv, json, bigdata, database] 
## read data, raw data, data clening, data analysis, feature engineering

## problem statement : "1. Get all NAN features from data.."
## "2. Getting datatypes of features & its overview"


# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


df = pd.read_csv("E:/Udemy Learning/Zomato Use Case/zomato.csv")


# In[9]:


df.head()


# In[11]:


df.shape


# In[12]:


## here columns like url, address, name, etc are "features" .


# In[13]:


df.columns


# In[15]:


df.isnull().sum()


# In[10]:


## for get NAN feature according loops
feature_nan = []
for feature in df.columns:
    if df[feature].isnull().sum()>1:
        feature_nan.append(feature)
feature_nan

## output of our 1st problem statement


# In[18]:


## for get NAN feature acording block of code
[feature for feature in df.columns if df[feature].isnull().sum()>1]

## output of our 1st problem statement


# In[7]:


df = pd.read_csv("E:/Udemy Learning/Zomato Use Case/zomato.csv")


# In[8]:


df['rate'].isnull().sum() ## get particuler column data


# In[9]:


df['rate'].isnull().sum()/len(df)*100


# In[15]:


for feature in feature_nan:
    print(' {} has {} % missing value '.format(feature,df[feature].isnull().sum()/len(df)*100))


# In[16]:


for feature in feature_nan:
    print(' {} has {} % missing values '.format(feature, np.round(df[feature].isnull().sum()/len(df)*100,3)))


# In[17]:


df.info()  
## output of second probelm statement that is overview of the features


# In[ ]:




