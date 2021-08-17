#!/usr/bin/env python
# coding: utf-8

# In[1]:


## read this file-manipulated file
## real world scenario - raw data [csv, json, bigdata, database] 
## read data, raw data, data clening, data analysis, feature engineering

## problem statement : "1. Get all NAN features from data.."
## "2. Getting datatypes of features & its overview"


# In[42]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[43]:


df = pd.read_csv("E:/Udemy Learning/Zomato Use Case/zomato.csv")


# In[44]:


df.head()


# In[45]:


df.shape


# In[12]:


## here columns like url, address, name, etc are "features" .


# In[46]:


df.columns


# In[47]:


df.isnull().sum()


# In[48]:


## for get NAN feature according loops
feature_nan = []
for feature in df.columns:
    if df[feature].isnull().sum()>1:
        feature_nan.append(feature)
feature_nan

## output of our 1st problem statement


# In[49]:


## for get NAN feature acording block of code
[feature for feature in df.columns if df[feature].isnull().sum()>1]

## output of our 1st problem statement


# In[50]:


df = pd.read_csv("E:/Udemy Learning/Zomato Use Case/zomato.csv")


# In[51]:


df['rate'].isnull().sum() ## get particuler column data


# In[52]:


df['rate'].isnull().sum()/len(df)*100


# In[53]:


for feature in feature_nan:
    print(' {} has {} % missing value '.format(feature,df[feature].isnull().sum()/len(df)*100))


# In[54]:


for feature in feature_nan:
    print(' {} has {} % missing values '.format(feature, np.round(df[feature].isnull().sum()/len(df)*100,3)))


# In[4]:


df.info()  
## output of second probelm statement that is overview of the features


# In[55]:


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


# In[56]:


df[df['approx_cost(for two people)'].isnull()]


# In[13]:


df['approx_cost(for two people)'].isnull().sum()


# In[14]:


df['approx_cost(for two people)'].unique()


# In[57]:


## remove comma from above unique output
## first we create a function which remove comma

def remove_comma(x):
    return x.replace(',','')

df['approx_cost(for two people)'].astype(str).apply(remove_comma)


# In[58]:


## now reduce the line of code using "Lambda" function
df['approx_cost(for two people)'].astype(str).apply(lambda x:x.replace(',',''))


# In[59]:


df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).apply(lambda x:x.replace(',',''))


# In[60]:


df['approx_cost(for two people)'].unique()


# In[61]:


df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(float)


# In[62]:


df['approx_cost(for two people)']

## output of 1st probelm statement of session 2


# In[63]:


df['approx_cost(for two people)'].dtype


# In[ ]:





# In[64]:


df['rate'].dtype


# In[65]:


df['rate'].unique()


# In[66]:


df['rate'].isnull().sum()


# In[67]:


## sample
print(df['rate'][0].split('/'))
print(df['rate'][0].split('/')[0])


# In[68]:


def split(x):
    return x.split('/')[0]
df['rate'] = df['rate'].astype(str).apply(split)


# In[69]:


df['rate'].unique()


# In[70]:


## still rate data has some dirty data like; '-' and 'NEW'
## for remove that type of data replace it

df['rate'].replace('-',0, inplace=True)
df['rate'].replace('NEW',0, inplace=True)


# In[71]:


df['rate'] = df['rate'].astype(float)


# In[72]:


df['rate'].unique()

## output of 2nd problem statement of 2nd session


# In[73]:


df['rate'].dtype


# In[ ]:





# In[74]:


## session 3 : problem stat1 : how many types of restorents we have ?

df['rest_type'].value_counts()


# In[75]:


## converting it into the bar graph

df['rest_type'].value_counts().nlargest(25).plot.bar(color='green')


# In[76]:


## create some amazing thing

plt.figure(figsize=(14,6))
df['rest_type'].value_counts().nlargest(20).plot.bar(color='red')


# In[77]:


## add new feature Top-types of restorents

def mark(x):
    if x in ('Quick Bites', 'Casual Dining'):
        return 'Quick bites + Casual dining'
    else:
        return 'others'
    
df['top_types'] = df['rest_type'].apply(mark)


# In[78]:


df.head()


# In[79]:


df['top_types'].value_counts()


# In[80]:


df['top_types'].value_counts().values


# In[81]:


df['top_types'].value_counts().index


# In[ ]:





# In[82]:


label = df['top_types'].value_counts().index
label


# In[83]:


value = df['top_types'].value_counts().values
value


# In[88]:


get_ipython().system('pip install plotly')


# In[89]:


import plotly.express as px


# In[90]:


fig = px.pie(df, names=label, values=value)


# In[91]:


fig.show()


# In[ ]:





# In[92]:


df.columns

## session 4 problem statements
## create a new dataframe in which we have votes,cost, & rating of each restorents
## restoents overview analysis


# In[93]:


df.dtypes


# In[94]:


df.groupby('name').agg({'votes':'sum','url':'count'})


# In[ ]:





# In[95]:


df.groupby('name').agg({'votes':'sum', 'url':'count', 'approx_cost(for two people)':'mean', 'rate':'mean'})


# In[104]:


rest = df.groupby('name').agg({'votes':'sum', 'url':'count', 'approx_cost(for two people)':'mean', 'rate':'mean'}).reset_index()


# In[97]:


rest


# In[105]:


rest.columns=['name', 'total_votes', 'total_unities', 'avg_approx_cost', 'avg_rate']


# In[106]:


rest


# In[107]:


rest.head()


# In[108]:


rest['votes_per_unity'] = rest['total_votes']/rest['total_unities']


# In[109]:


rest.head()


# In[111]:


popular = rest.sort_values(by="total_unities",ascending=False)


# In[112]:


popular
## output of 1st problem statement of session 4
## "popular" is our new datafram which satisfies 1st prob staement.


# In[114]:


popular.shape


# In[121]:


popular.nunique()


# In[122]:


popular['name'].nunique()


# In[152]:


data=popular.sort_values(by='total_votes', ascending=False).query('total_votes>0').head(5)
data
data=popular.sort_values(by='total_votes', ascending=False).query('total_votes>0').tail(5)
data


# In[155]:


## 2nd problem statement : overview analysis of restorent
## avarage votes recived by restorent
## top 5 most votes resto
## top 5 less votes resto

import seaborn as sns
fig, (ax1,ax2,ax3) =plt.subplots(3, 1, figsize=(10,20))
ax1.text(0.5, 0.5, int(popular['total_votes'].mean()), fontsize=40, ha='center')
ax1.text(0.5,0.3, 'Avarage votes', fontsize=25, ha='center')
ax1.text(0.5,0.1, 'received by restorents', fontsize=25, ha='center')

sns.barplot(x='total_votes', y='name', data=popular.sort_values(by='total_votes', ascending=False).query('total_votes>0').head(5), ax=ax2, palette='plasma')
ax2.set_title('top 5 most rated restorents')

sns.barplot(x='total_votes', y='name', data=popular.sort_values(by='total_votes', ascending=False).query('total_votes>0').tail(5), ax=ax3, palette='plasma')
ax3.set_title('top 5 least rated restorents')




## solution of prob statement 2 of session 4


# In[ ]:




