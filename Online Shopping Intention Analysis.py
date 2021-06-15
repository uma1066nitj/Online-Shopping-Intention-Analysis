#!/usr/bin/env python
# coding: utf-8

# In[21]:


get_ipython().system(' pip install plotly==4.14.3')


# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py


# In[8]:


df = pd.read_csv('online_shoppers_intention.csv')


# In[9]:


df.head()


# finding the missing values and fill the missing place with the help of fillna

# In[10]:


missing = df.isnull().sum()
missing


# In[11]:


df.fillna(0,inplace=True)


# In[12]:


df.head()


# In[13]:


x= df.iloc[:,[5,6]].values


# In[14]:


x


# In[15]:


from sklearn.cluster import KMeans


# In[16]:


wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i,
               init = 'k-means++',
               max_iter=300,
               random_state=0,
               n_init = 10,
               algorithm='full',
               tol=0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)

plt.rcParams['figure.figsize'] = (13,7)
plt.plot(range(1,11),wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method',fontsize=20)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.show()


# In[17]:


km = KMeans(n_clusters=2,init = 'k-means++',max_iter=300,n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means==0,0],x[y_means==0,1],s=50,c='yellow',label='Uninterested Customer')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=50,c='green',label='Target Customer')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=50,c='blue',label='centroid')
plt.title('ProductRelated Duration vs Bounce Rate',fontsize=15)
plt.xlabel("ProductRelated Duration")
plt.ylabel("Bounce Rate")
plt.legend()
plt.show()


# In[18]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels_true = le.fit_transform(df['Revenue'])

#get predicted clustering result label
labels_pred = y_means

#print adjusted rand index, which measures the similarity of the two assignments
from sklearn import metrics
score = metrics.adjusted_rand_score(labels_true,labels_pred)
print('Adjusted rand index: ')
print(score)


# In[3]:


get_ipython().system(' pip install scikit-plot')


# In[19]:


import scikitplot as skplt
plt_1 = skplt.metrics.plot_confusion_matrix(labels_true,labels_pred,normalize=False)
plt_2 = skplt.metrics.plot_confusion_matrix(labels_true,labels_pred,normalize=True)


# In[ ]:




