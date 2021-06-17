#!/usr/bin/env python
# coding: utf-8

# KNN - Predict whether a review is positive or nagetive

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import nltk
from nltk.corpus import stopwords
import re
from heapq import nlargest
from sklearn.model_selection import KFold


trainPath='train_data.txt'
train1,train2=[],[]
test1,test2=[],[]
accuracyAll=0.0
errors=0


# In[2]:


def processTrain(path):
    with open(path,'r') as f:
        lines=f.readlines()[1:]
    #read data from train_data.txt with lowercase
    for line in lines:
        if line[0] is '-':
            train1.append([-1,line[3:].lower()])
        else:
            train1.append([1,line[3:].lower()])
        train2.append(line[3:].lower())
    f.close()
    #remove all punctuations and special symbols in the reviews
    for i in range(len(train1)):
        train1[i][1]=re.sub(r'</?\w+[^>]*>|[^\w\s]','',train1[i][1])
        train2[i]=re.sub(r'</?\w+[^>]*>|[^\w\s]','',train2[i])
    print(len(train1))
    return train1,train2


# In[3]:

#read data from test_data.txt with lowercase
def processTest(path):
    with open(path,'r') as f:
        lines=f.readlines()
    for line in lines:
        #initialize all test reviews' classification as 0
        test1.append([0,line.lower()])
        test2.append(line.lower())
    f.close()
    for i in range(len(test1)):
        test1[i][1]=re.sub(r'</?\w+[^>]*>|[^\w\s]','',test1[i][1])
        test2[i]=re.sub(r'</?\w+[^>]*>|[^\w\s]','',test2[i])
    print(len(test1))
    return test1,test2


# In[4]:


def tfidf(tr_x,tr_y,te_x,te_y):
    global trTfidf,teTfidf
    vectorizer=TfidfVectorizer(max_features=500,stop_words='english',ngram_range=(1,1))
    #build a term-dsocument matrix
    trTfidf=vectorizer.fit_transform(tr_y)
    #add the original classification to the matrix as a column
    classLabel=[]
    for i in range(len(tr_x)):
        classLabel.append([tr_x[i][0]])
    classification= np.array(classLabel)
    np.vstack(classification)
    trTfidf = scipy.sparse.hstack([trTfidf, classification])
    #assign random number to teat reviews' classification
    teTfidf=vectorizer.transform(te_y)
    randomClass = np.ones((teTfidf.shape[0],1))
    teTfidf=scipy.sparse.hstack([teTfidf, randomClass])
    return trTfidf,teTfidf


# In[5]:


def knn(tr_x,te_x,k):
    dist=[]
    knearest=[]
    j=0
    #caculate cosine distance between two vectors
    for v1 in teTfidf.toarray():
        for v2 in trTfidf.toarray():
            v1=v1[0:teTfidf.shape[1]-2]
            v2=v2[0:trTfidf.shape[1]-2]
            dist.append(cosine_distance(v1,v2))
        #find k nearest neighbor
        knearest=list(map(dist.index,nlargest(k,(dist))))
        negative=0
        positive=0 
        knearest.reverse()
        for i in range(len(knearest)):
            #knearest[i] is the ith reviews
            if trMtr[knearest[i],trTfidf.shape[1]-1]==-1:
                #weighted 1*(i+1)
                negative=negative+1*(i+1)
            else:
                positive=positive+1*(i+1)
        if negative>positive:
            te_x[j][0]=-1
        else:
            te_x[j][0]=1
        del knearest[:]
        del dist[:]
        j=j+1
    #write out the predicted result in a file
    fw=open('result_f.txt','w')
    for i in range(len(test1)):
        fw.write(str(te_x[i][0]))
        fw.write('\n')
    fw.close()
    return te_x


# In[6]:

#cosine distance
def cosine_distance(v1,v2):
    return np.dot(v1,v2)/(np.sqrt(np.sum(v1**v2))*np.sqrt(np.sum(v2**2)))


# In[7]:

#use cross validation compare the accuracy of different k
def cross_validation():
    acc=0.0
    crTrain1=[]
    crTrain2=[]
    crTest1=[]
    crTest2=[]
    correct=[]
    global errors,accuracyAll
    train1,train2=processTrain(trainPath)
    #implement cross validation use kfold
    kf=KFold(n_splits=5,shuffle=True,random_state=None)
    for k in (23,423,2233,4133):
        for train_index,test_index in kf.split(train1):
            #print(train_index,test_index)
            for i in train_index:
                crTrain1.append(train1[i])
                crTrain2.append(train1[i][1])
            for i in test_index:
                crTest1.append([0,train1[i][1]])
                crTest2.append(train1[i][1])
                correct.append(train1[i])
            preTest=main(crTrain1,crTrain2,crTest1,crTest2,k)
            get_acc(preTest,correct)
            del crTrain1[:]
            del crTrain2[:]
            del crTest1[:]
            del crTest2[:]
            del correct[:]
        print("K= "+ str(k)+ "  accuracy= "+ str(accuracyAll/5))
        print("errors= "+ str(errors))
        errors = 0
        accuracyAll=0.




# In[8]:


def get_acc(x,y):
    acc=0.0
    global errors,accuracyAll
    i=0
    for j in range(len(y)):
        if x[i][0]==y[j][0]:
            acc=acc+1
        else:
            errors=errors+1
        j=j+1
    accuracyAll=(acc/len(x))+accuracyAll
    print("accuracy: " + str(acc/len(x)))


# In[9]:


def main(tr_x,tr_y,te_x,te_y,k):
    trTfidf,teTfidf=tfidf(tr_x,tr_y,te_x,te_y)
    global trMtr,teMtr
    trMtr=trTfidf.tocsr()
    teMtr=teTfidf.tocsr()
    return knn(tr_x,te_x,k)


# cross_validation()

# In[ ]:


tr_x,tr_y=processTrain(trainPath)
testPath='test_data.txt'
te_x,te_y=processTest(testPath)
#np.seterr(invalid='ignore')
main(tr_x,tr_y,te_x,te_y,423)
