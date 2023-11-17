#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


# In[3]:


from sklearn.model_selection import train_test_split


# In[10]:


data=pd.read_csv("housing.csv")
data


# In[11]:


data.info()


# In[12]:


#Few null value in ->  4   total_bedrooms      20433 non-null  float64
data.dropna(inplace=True) #drops nan value row
data.info()


# In[13]:


x=data.drop(['median_house_value'],axis=1)
y=data['median_house_value']


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train


# In[15]:


train_data=x_train.drop(['ocean_proximity'],axis=1,inplace=True)
train_data


# In[16]:


train_data=x_train.join(y_train)
train_data


# In[17]:


train_data.hist(figsize=(15,8))


# In[18]:


#correlation
#train_data.corr()
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap="PiYG")


# In[19]:


#right or left skwed dataset
train_data['total_rooms']=np.log(train_data['total_rooms']+1)
train_data['total_bedrooms']=np.log(train_data['total_bedrooms']+1)
train_data['population']=np.log(train_data['population']+1)
train_data['households']=np.log(train_data['households']+1)
train_data.hist(figsize=(15,8))


# In[20]:


#train_data.drop([9],axis=1,inplace=True)
train_data.columns.values


# In[22]:


#adding ocean_proximity column to train_data and considering it as factor to determine housing price
if 'ocean_proximity' not in train_data.columns.values:
    train_data['ocean_proximity']=data['ocean_proximity']
train_data.ocean_proximity.value_counts()


# In[23]:


#hard encoding-pd.get_dummies(train_data.ocean_proximity)
#making column for each label as t/f
train_data=train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop('ocean_proximity',axis=1)


# In[24]:


train_data.head()


# In[25]:


#correlation
#train_data.corr()
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap="PiYG")


# In[26]:


plt.figure(figsize=(15,8))
sns.scatterplot(x="latitude",y="longitude",data=train_data,hue="median_house_value", palette="coolwarm")


# In[27]:


#two other feature no. of bedrooms and nearing household rooms
train_data['bedroom_ratio']=train_data['total_bedrooms']/train_data['total_rooms']
train_data['household_rooms']=train_data['total_rooms']/train_data['households']


# In[28]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap="PiYG")


# In[30]:


#Split test data with new features
from sklearn.linear_model import LinearRegression
x_train,y_train=train_data.drop(['median_house_value'],axis=1),train_data['median_house_value']
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[31]:


#evaluation 
test_data=x_train.join(y_test)
#right or left skwed dataset
test_data['total_rooms']=np.log(test_data['total_rooms']+1)
test_data['total_bedrooms']=np.log(test_data['total_bedrooms']+1)
test_data['population']=np.log(test_data['population']+1)
test_data['households']=np.log(test_data['households']+1)

#two other feature no. of bedrooms and nearing household rooms
test_data['bedroom_ratio']=test_data['total_bedrooms']/test_data['total_rooms']
test_data['household_rooms']=test_data['total_rooms']/test_data['households']


# In[32]:


x_test,y_test=test_data.drop(['median_house_value'],axis=1),test_data['median_house_value']
x_test.shape


# In[33]:


res=reg.score(x_train,y_train)
print("The Accuracy acquired with selected feature:",res*100)


# In[ ]:




