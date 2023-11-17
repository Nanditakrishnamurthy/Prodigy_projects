#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("Mall_Customers.csv")
df.head(10)


# In[24]:


df.shape


# In[25]:


df.info()


# In[26]:


x=df.iloc[:,[3,4]].values
x


# In[27]:


from sklearn.cluster import KMeans
wcss=[]


# In[28]:


for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[29]:


#finding optimal no. of clusters
plt.plot(range(1,11),wcss)
plt.title("Elbow")
plt.xlabel("Clusters")
plt.ylabel("WCSS Values")
plt.show()


# In[30]:


kmeansmodel=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans=kmeansmodel.fit_predict(x)


# In[32]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=80,c='pink',label="Customer 1")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=80,c='black',label="Customer 2")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=80,c='blue',label="Customer 3")
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=80,c='red',label="Customer 4")
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=80,c='lavender',label="Customer 5")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label='Centroids')
plt.title("Customer Cluster")
plt.xlabel('Annual Income (k$)')
plt.ylabel("Spending score")
plt.legend()
plt.show()


# In[ ]:




