#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import re
import scipy
import math
from copy import deepcopy
from random import randint
from sklearn.cluster import KMeans
import random
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
import matplotlib.pyplot as plt

data_path='./1601384482_8387134_image_new_test.txt'
iris_path='./1601384279_9602122_iris_new_data.txt'
test=[]

N=10000
FEATURES=784
MAX_ITER=300
VMEASURE=10


# In[2]:


def read_image_data():
    global test
    #read data from .txt file to numpy array
    test=np.loadtxt(data_path,delimiter = ",")
    #feature normalization
    test=test.astype('float32')
    test/=255


# In[3]:


def read_iris_data():
    global test
    test=np.loadtxt(iris_path)
    #feature normalization
    test=normalize(test)


# In[4]:


def km_plus_plus(test, K):
    c=randint(0,N-1)
    centroids = [test[c]]
    for _ in range(1, K):
        dist = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in test])
        probs = dist/dist.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        centroids.append(test[i])
    return centroids


# In[5]:


def kmeans(test,K):
    #iterate kmean algorithm and record cluster for each instance
    runs=0
    clusters=np.zeros((N,VMEASURE))
    sil_vals=np.zeros(VMEASURE)
    sse_vals=np.zeros(VMEASURE)
    while runs<VMEASURE:
        old_centroids=np.zeros((K,FEATURES))
        #initialize cluster centroids by kmeans++ algorithm
        centroids=km_plus_plus(test,K)
        new_centroids=np.array(centroids)
        #record distance between each point with centroids
        dist = np.zeros((N,K))
        j=0
        while j<MAX_ITER:
            print('there is '+str(j))
            for row in range(N):
                for i in range(K):
                    #calculate distance between each point and centroids
                    dist[row][i]=correlation(test[row],new_centroids[i])
#                     dist[row][i]=consine_sim(test[row],new_centroids[i])
#                     dist[row][i]=euclidean(test[row],new_centroids[i])
            #record the point into the closet centriod set
            clusters[:,runs]=np.argmax(dist,axis=1)
            old_centroids=deepcopy(new_centroids)
            #recalculate k centroid points
            for i in range(K):
                new_centroids[i]=np.mean(test[clusters[:,runs] == i],axis =0)
            #check if the centriods have no change
            if np.all(old_centroids==new_centroids):
                break
            j+=1
        sil_vals[runs]=silhouette(test,clusters[:,runs])
        sse1=SSE(test,clusters[:,runs],new_centroids,K)
        sse_vals[runs]=sse1
        print("sil_vals " + str(runs) + ": " + str(sil_vals[runs]) )
        print("sse_vals " + str(runs) + ": " + str(sse_vals[runs]) )
        runs+=1
    #get the 1st best cluster with highest silhouette score
    highest_sil=clusters[:,np.argmax(sil_vals)]
    #get the 2nd best cluster with minimum sse 
    min_sse=clusters[:,np.argmin(sse_vals)]
    print("itr: "+ str(np.argmax(sil_vals)) + " has the maximum silhouette score = " + str(np.amax(sil_vals)))
    print("itr: "+ str(np.argmin(sse_vals)) + " has the minimum sse  = " + str(np.amin(sse_vals)))
    with open('img_sil.txt','w') as f:
        for i in highest_sil:
            f.write(str((int(i)+1)))
            f.write('\n')
    with open('img_sse.txt','w') as f:
        for i in min_sse_vals:
            f.write(str((int(i)+1)))
            f.write('\n')
    return np.amin(sse_vals)


# In[6]:


def k_means(test,K):
    old_centroids=np.zeros((K,FEATURES))
    #initialize cluster centroids by kmeans++ algorithm
    centroids=km_plus_plus(test,K)
    new_centroids=np.array(centroids)
    dist = np.zeros((N,K))
    j=0
    while j<MAX_ITER:
        print('there is '+str(j))
        for row in range(N):
            for i in range(K):
                dist[row][i]=correlation(test[row],new_centroids[i])
#                     dist[row][i]=consine_sim(test[row],new_centroids[i])
#                     dist[row][i]=euclidean(test[row],new_centroids[i])
        labels=np.argmax(dist,axis=1)
        old_centroids=deepcopy(new_centroids)
        for i in range(K):
            new_centroids[i]=np.mean(test[labels == i],axis =0)
        if np.all(old_centroids==new_centroids):
            break
        j+=1
    sil_vals=silhouette(test,labels)
    sse1=SSE(test,labels,new_centroids,K)
    sse_vals=sse1
    print("sil_vals " + str(K) + ": " + str(sil_vals) )
    print("sse_vals " + str(K) + ": " + str(sse_vals) )
    return np.amin(sse_vals)


# In[7]:


def SSE(test,clusters,centers,K):
    dist=np.zeros(test.shape[0])
    for k in range(K):
        dist[clusters==k]=np.linalg.norm(test[clusters==k]- centers[k], axis=1)
        dist=np.square(dist)
        sse=np.sum(dist)
        return sse


# In[8]:


def silhouette(data,predicted):
    score = silhouette_score(data, predicted, metric='correlation')
    return score


# In[9]:


def correlation(v1,v2):
    corr=np.corrcoef(v1,v2)
    return corr[0][1]


# In[10]:


def euclidean(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))


# In[11]:


def consine_sim(v1,v2):
    return np.dot(v1,v2)/(np.sqrt(np.sum(v1**2))*np.sqrt(np.sum(v2**2)))


# In[12]:


def plot_sse(sse):
    plt.figure()
    filename="SUM OF SQUARED ERROR"
    x=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    plt.plot(x, sse,'-o')
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.savefig(filename)
    plt.show()


# In[13]:


def main():
    read_image_data()
#     read_iris_data()
#     sse=[]
#     for K in range(2,20,1):
#         sse.append(k_means(test,K))
#         print(sse)
#     plot_sse(sse)
    kmeans(test,8)


# In[14]:


main()

