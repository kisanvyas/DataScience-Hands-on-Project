#!/usr/bin/env python
# coding: utf-8

# In[1]:


## read this file-manipulated file
## real world scenario - raw data [csv, json, bigdata, database] 
## read data, raw data, data clening, data analysis, feature engineering

## problem statement : "1. Get all NAN features from data.."
## "2. Getting datatypes of features & its overview"


# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


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


# In[4]:


df.info()  
## output of second probelm statement that is overview of the features


# In[5]:


## session 2 : problem statments : 1st : perform data cleaning on feature - approx_cost
## 2nd : clean rate_num column



df['approx_cost(for two people)'].dtype


# In[6]:


df['rate'].dtype


# In[7]:


df['votes'].dtype


# In[8]:


df['approx_cost(for two people)'].isnull()


# In[9]:


df['votes'].isnull()


# In[12]:


df[df['approx_cost(for two people)'].isnull()]


# In[13]:


df['approx_cost(for two people)'].isnull().sum()


# In[14]:


df['approx_cost(for two people)'].unique()


# In[16]:


## remove comma from above unique output
## first we create a function which remove comma

def remove_comma(x):
    return x.replace(',','')

df['approx_cost(for two people)'].astype(str).apply(remove_comma)


# In[18]:


## now reduce the line of code using "Lambda" function
df['approx_cost(for two people)'].astype(str).apply(lambda x:x.replace(',',''))


# In[19]:


df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).apply(lambda x:x.replace(',',''))


# In[20]:


df['approx_cost(for two people)'].unique()


# In[21]:


df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(float)


# In[22]:


df['approx_cost(for two people)']

## output of 1st probelm statement of session 2


# In[24]:


df['approx_cost(for two people)'].dtype


# In[ ]:





# In[25]:


df['rate'].dtype


# In[26]:


df['rate'].unique()


# In[27]:


df['rate'].isnull().sum()


# In[29]:


## sample
print(df['rate'][0].split('/'))
print(df['rate'][0].split('/')[0])


# In[30]:


def split(x):
    return x.split('/')[0]
df['rate'] = df['rate'].astype(str).apply(split)


# In[31]:


df['rate'].unique()


# In[33]:


## still rate data has some dirty data like; '-' and 'NEW'
## for remove that type of data replace it

df['rate'].replace('-',0, inplace=True)
df['rate'].replace('NEW',0, inplace=True)


# In[34]:


df['rate'] = df['rate'].astype(float)


# In[35]:


df['rate'].unique()

## output of 2nd problem statement of 2nd session


# In[36]:


df['rate'].dtype


# In[ ]:





# In[38]:


## session 3 : problem stat1 : how many types of restorents we have ?

df['rest_type'].value_counts()


# In[41]:


## converting it into the bar graph

df['rest_type'].value_counts().nlargest(25).plot.bar(color='green')


# In[44]:


## create some amazing thing

plt.figure(figsize=(14,6))
df['rest_type'].value_counts().nlargest(20).plot.bar(color='red')


# In[9]:


## add new feature Top-types of restorents

def mark(x):
    if x in ('Quick Bites', 'Casual Dining'):
        return 'Quick bites + Casual dining'
    else:
        return 'others'
    
df['top_types'] = df['rest_type'].apply(mark)


# In[47]:


df.head()


# In[48]:


df['top_types'].value_counts()


# In[50]:


df['top_types'].value_counts().values


# In[51]:


df['top_types'].value_counts().index


# In[ ]:





# In[12]:


label = df['top_types'].value_counts().index
label


# In[11]:


value = df['top_types'].value_counts().values
value


# In[6]:


get_ipython().system('pip install plotly')


# In[7]:


import plotly.express as px


# In[17]:


fig = px.pie(df, names=label, values=value)


# In[18]:


fig.show()


# In[ ]:




