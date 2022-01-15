#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation: Data Science and Business Analytics Internship
# 
# ## Task 1: Prediction using Unsupervised ML
# ### Predict the optimum number of clusters and represent it visually from the given dataset.
# 
# ## Author: Jankee Khandivar
# ### Batch: December - 2021

# #### Dataset URL: https://bit.ly/3kXTdox

# ## Step 1: Data Pre-processing 

# ### Importing the libraries

# In[4]:


import numpy as np                # for mathematical calculation
import pandas as pd               # for managing the dataset
import matplotlib.pyplot as plt   # for plotting the graph
from sklearn.cluster import KMeans


# ### Importing the dataset

# In[2]:


dataset = pd.read_csv("F:\Spark foundation\Task_2_data.csv")
print(dataset.shape)
dataset.head(10)


# ### Extracting Independent Variables

# In[3]:


x = dataset.iloc[:, [0,1,2,3]].values


# ## Step 2: Finding the optimum number of clusters

# ### Initialising the list for the values of wcss

# In[9]:


wcss_list = []       


# ### Using for loop for iterations from 1 to 10

# In[10]:


for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init = 10, random_state=0, max_iter=450)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)


# ### Plotting a result onto a line graph

# In[11]:


plt.plot(range(1,11), wcss_list)
plt.title("The Elbow method Graph")
plt.xlabel("Number of clusters")
plt.ylabel("wcss_list")
plt.show()


# ## Step 3: Training the K-Means algorithm on the taining dataset

# In[13]:


kmeans = KMeans(n_clusters=3, init='k-means++',random_state=0)
y_pred = kmeans.fit_predict(x)


# ## Step 4: Visualizing the clusters

# In[14]:


# for first cluster
plt.scatter(x[y_pred == 0,0], x[y_pred==0,1], s=75, c='blue', label='Cluster_1')

# for second cluster
plt.scatter(x[y_pred == 1,0], x[y_pred == 1,1], s=75, c='green', label='Cluster_2')

# for third cluster
plt.scatter(x[y_pred == 2,0], x[y_pred == 2,1], s=75, c='red', label='Cluster_3')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=150, c='yellow', label='centroid')

plt.title("clusters")
plt.legend()
plt.show()

