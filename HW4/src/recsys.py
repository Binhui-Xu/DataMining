#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy as sp
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.stats
import scipy.spatial
import operator
from math import sqrt
import math
     
movie_actors_path = './additional_files/movie_actors.dat'
movie_directors_path = './additional_files/movie_directors.dat'
movie_genres_path='./additional_files/movie_genres.dat'
movie_tags_path='./additional_files/movie_tags.dat'
tags_path='./additional_files/tags.dat'
test_path='./additional_files/test.dat'
train_path='./additional_files/train.dat'
user_taggedmovies_path='./additional_files/user_taggedmovies.dat'


# In[2]:


def read_data():
    global movie_actors,movie_directors,movie_genres,movie_tags,tags,user_taggedmovies
    movie_actors=pd.read_table(movie_actors_path,sep = '\t',header=0,engine='python',error_bad_lines=False)
    movie_directors=pd.read_table(movie_directors_path,sep = '\t',header=0,engine='python',error_bad_lines=False)
    movie_genres=pd.read_table(movie_genres_path,sep = '\t',header=0,engine='python',error_bad_lines=False)
    movie_tags=pd.read_table(movie_tags_path,sep = '\t',header=0,engine='python',error_bad_lines=False)
    tags=pd.read_table(tags_path,sep = '\t',header=0,engine='python',error_bad_lines=False)
    test=pd.read_table(test_path,sep = ' ',header=0,engine='python',error_bad_lines=False)
    train= pd.read_table(train_path,sep = ' ',header=0,engine='python')
    user_taggedmovies=pd.read_table(user_taggedmovies_path,sep = ' ',header=0,engine='python',error_bad_lines=False)

    return train,test


# In[3]:


def preprocess(rate):
    rate_norm=rate.apply(lambda x : (x-np.mean(x))/(np.max(x)-np.min(x)),axis=1)
    rate_norm.fillna(0,inplace=True)
    rate_norm=rate_norm.T
    rate_sparse=sp.sparse.csr_matrix(rate_norm.values)
    return rate_norm,rate_sparse


# In[4]:


def get_movie_details():
    #get top 3 actors for each movie
    top3actors=movie_actors.sort_values(['movieID','ranking'],ascending=True).groupby('movieID').head(3)
    actors=top3actors.drop(['actorID','ranking'],axis=1)
    actors = actors.groupby(by='movieID').apply(lambda x:[','.join(x['actorName'])])
    actorsdf=pd.DataFrame(actors,columns=['actors'])
    actorsdf.reset_index(inplace=True)

    #get director for each movie
    directors = movie_directors.groupby(by='movieID').apply(lambda x:[','.join(x['directorName'])])
    directorsdf=pd.DataFrame(directors,columns=['directors'])
    directorsdf.reset_index(inplace=True)
    
    #get genres for each movie
    genres = movie_genres.groupby(by='movieID').apply(lambda x:[','.join(x['genre'])])
    genredf=pd.DataFrame(genres,columns=['genres'])
    genredf.reset_index(inplace=True)
    
    #get tags for each movie
    tags2=pd.merge(user_taggedmovies,tags,left_on=['tagID'],right_on=['id']).drop(['id'],axis=1)
    tagsall=tags2.groupby(by='movieID').apply(lambda x:[','.join(x['value'])])
    tagsdf=pd.DataFrame(tagsall,columns=['tags'])
    tagsdf.reset_index(inplace=True)
    
    #get user tags for each movie
    tags3=pd.merge(movie_tags,tags,left_on=['tagID'],right_on=['id']).drop(['id'],axis=1)
    usertags=tags3.groupby(by='movieID').apply(lambda x:[','.join(x['value'])])
    usertagsdf=pd.DataFrame(usertags,columns=['usertags'])
    usertagsdf.reset_index(inplace=True)
    
    #merge all movie details together
    movies=reduce(lambda x,y: pd.merge(x,y, on='movieID', how='outer'), [actorsdf,directorsdf,genredf,tagsdf,usertagsdf])
    return movies


# In[5]:


def clean_data(x):
    #convert all strings to lower case and strip names of spaces
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[6]:


def merge_details(x):
    return ' '.join(x['actors']) + ' ' + ' '.join(x['directors']) + ' ' +' '.join(x['genres']) + ' ' + ' '.join(x['tags'])+ ' ' + ' '.join(x['usertags'])


# In[7]:


def tfidf(movies):
    tfidf=TfidfVectorizer(stop_words='english')
    movies['details']=movies['details'].fillna('')
    tr_tfidf=tfidf.fit_transform(movies['details'])
    print(tr_tfidf.shape)
    return tr_tfidf


# In[8]:


def cross_validation(train,movies):
    rmse=0.0
    actual=[]
    rating_col=train.loc[:,['rating']]
    kf=KFold(n_splits=10,shuffle=True,random_state=None)
    for k in (25,455,1655):
        for train_index,test_index in kf.split(train):
            train_cv=train.loc[train_index,:]
            test_cv=train.loc[test_index,:].drop('rating',axis=1)
            actual=rating_col.iloc[test_index].values
#             predict=correlative(train_cv,test_cv,k)
            predict=content_based(train_cv,test_cv,movies,k)
            rmse+=get_rmse(predict,actual)
        print("K= "+ str(k)+ "  rmse= "+ str(rmse/10))
        rmse=0


# In[9]:


def content_based(traindf,testdf,movies,k):
    rate_pivot=traindf.pivot_table(index=['userID'],columns=['movieID'],values='rating')
    tr_tfidf=tfidf(movies)
    movie_cos_sim=cosine_similarity(tr_tfidf,tr_tfidf) 
    movies = movies.reset_index()
    indices = pd.Series(movies.index, index=movies['movieID'])
    result=[]
    for i,r in tqdm(testdf.iterrows()):
        userid=testdf.loc[i]['userID']
        movieid=testdf.loc[i]['movieID']
        #get top movies
        # Get the index of the movie that matches the movieid
        idx = indices[movieid]
        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(movie_cos_sim[idx]))
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the scores of the k most similar movies
        sim_scores = sim_scores[1:k]
        movie_values=[i[1] for i in sim_scores]
        # Get the movie indices
        sim_movies = [i[0] for i in sim_scores]
        predict=cb_predictor(userid,movieid,sim_movies,movie_values,rate_pivot)
        result.append(predict)
    with open('cb-result.txt','w') as f:
        for i in result:
            f.write(str((int(i))))
            f.write('\n')
    return result


# In[10]:


def collaborative(traindf,testdf,k):
    print(traindf.head())
    print(testdf.head())
    n_users = traindf.userID.unique().shape[0]
    n_movies = traindf.movieID.unique().shape[0]
    rate_pivot=traindf.pivot_table(index=['userID'],columns=['movieID'],values='rating')
    rate_norm,rate_sparse=preprocess(rate_pivot)
    #caculate similairyty
    user_cos_sim,user_pearson=get_similarity(rate_norm,rate_sparse,n_users,n_movies)
    #put all similarity scores in one dataframe
    user_sim_df = pd.DataFrame(user_cos_sim,index = rate_norm.columns,columns = rate_norm.columns)
#     user_sim_df = pd.DataFrame(user_pearson,index = rate_norm.columns,columns = rate_norm.columns)
    #predict rating
    result=[]
    for i,r in testdf.iterrows():
        userid=testdf.loc[i]['userID']
        movieid=testdf.loc[i]['movieID']
        #get top neighbors
        # Get the user indices
        sim_users = user_sim_df.sort_values(by=userid, ascending=False).index[1:k]
        #Get the scores of the k most similar users 
        user_values = user_sim_df.sort_values(by=userid, ascending=False).loc[:,userid].tolist()[1:k]
        predict=cf_predictor(userid,movieid,sim_users,user_values,rate_pivot)
        result.append(predict)
    with open('cf-result.txt','w') as f:
        for i in result:
            f.write(str((int(i))))
            f.write('\n')
    return result


# In[11]:


def get_similarity(rate_norm,rate_sparse,n_users,n_movies):
    user_cos_sim=np.zeros((n_users,n_users))
    user_pearson=np.zeros((n_users,n_users))
    #caculate cosine similarity
    user_cos_sim = cosine_similarity(rate_sparse.T)
    #caculate pearson correlation
    for i in tqdm(range(n_users)):
        for j in range(n_users):
            if np.count_nonzero(rate_norm.iloc[i,:]) and np.count_nonzero(rate_norm.iloc[j,:]):
                try:
                    if not math.isnan(scipy.stats.pearsonr(rate_norm.iloc[i,:],rate_norm.iloc[j,:])[0]):
                        user_pearson[i][j]=scipy.stats.pearsonr(rate_norm.iloc[i,:],rate_norm.iloc[j,:])[0]
                    else:
                        user_pearson[i][j]=0
                except:
                    user_pearson[i][j]=0
    return user_cos_sim,user_pearson


# In[12]:


def cf_predictor(userid,movieid,sim_users,user_values,rate_pivot):
    #user-based correlative filtering
    rating_list = []
    weight_list = []
    predict=3.0
    for j, i in enumerate(sim_users):
        try:
            rating = rate_pivot.loc[i, movieid]
            similarity = user_values[j]
            if np.isnan(rating):
                continue
            elif not np.isnan(rating):
                rating_list.append(rating*similarity)
                weight_list.append(similarity)
        except KeyError:                                               
            pass  
    if (sum(weight_list)!=0):
        predict=round(sum(rating_list)/sum(weight_list))
    if predict<0:
        predict=0.0
    if predict>5:
        predict=5.0
    return predict


# In[13]:


def cb_predictor(userid,movieid,sim_movies,movie_values,rate_pivot):
    rating_list = []
    weight_list = []
    predict=3.0
    for j, i in enumerate(sim_movies):
        try:
            rating = rate_pivot.loc[userid,i]
            similarity = movie_values[j]
            if np.isnan(rating):
                continue
            elif not np.isnan(rating):
                rating_list.append(rating*similarity)
                weight_list.append(similarity)
        except KeyError:                                               
            pass  
    if (sum(weight_list)!=0):
        predict=round(sum(rating_list)/sum(weight_list))
    if predict<0:
        predict=0.0
    if predict>5:
        predict=5.0
    return predict


# In[14]:


def get_rmse(predict,actual):
    return sqrt(mean_squared_error(predict,actual))


# In[15]:


def main():
    train,test=read_data()
    movies=get_movie_details()
    attributes = ['actors', 'directors', 'genres', 'tags','usertags']
    for att in attributes:
        movies[att] = movies[att].apply(clean_data)
    movies['details']=movies.apply(merge_details,axis=1)
    train,test=read_data()
#     cross_validation(train,movies)
     collaborative(train,test,455)
#    content_based(train,test,movies,1655)
    


# In[16]:


main()

