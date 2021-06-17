#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold,RandomizedSearchCV,GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from scipy import interp
from scipy.stats import randint
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


clf_num = 5
y_test=[]
tr_path='./1600106342_882043_train.csv'
te_path='./1600106342_8864183_test.csv'


# In[2]:


#reading data from .csv files
def read_data():
    global train1,train2,test,credit
    #read training data from .csv file
    bank_data=pd.read_csv(tr_path)
    #categorical variables encoding--One-Hot encoding
    cat_features=['F3','F4','F7','F8','F9','F10','F11']
    num_features=['F1','F2','F5','F6','credit']
    for v in cat_features:
        dummies=pd.get_dummies(bank_data[cat_features])
        train1=pd.concat([dummies,bank_data[num_features]],axis=1)
    credit=train1['credit']
    train2=train1.drop(['credit'],axis=1)
    
    #read testing data from .csv file
    test_data=pd.read_csv(te_path)
    num_features=['F1','F2','F5','F6']
    for v in cat_features:
        dummies=pd.get_dummies(test_data[cat_features])
        test=pd.concat([dummies,test_data[num_features]],axis=1)
    


# In[3]:


#Synthetic Minority Oversampling Technique
#deal with imbalanced classes
def sampling(train_data,credit_data):
    smote=SMOTE(random_state=42)
    train_balanced,credit_balanced=smote.fit_resample(train_data,credit_data)
    return train_balanced,credit_balanced


# In[4]:


def cross_validation():  
    # array for results
    cv_result = np.zeros(shape=(clf_num,2))
    f1_score_arr = np.zeros(shape=(clf_num,10,2))
    #list of all classifier
    classifiers=[decision_tree,naive_bayes,random_forest,adaboost,SVM]
    #implement kfold cross validation,cv=10
    kf=KFold(n_splits=10,shuffle=True,random_state=None)
    i=0
    for train_index,test_index in kf.split(train2,credit):
        #split data from dataframe
        #dataframe.iloc[0,1,2,3] -- the 1st, 2nd, 3rd, 4th row 
        X_train, X_test, y_train, y_test = train2.iloc[train_index], train2.iloc[test_index], credit.iloc[train_index], credit.iloc[test_index]
        # oversampling X_train and y_train
        tr_bal,credit_bal=sampling(X_train,y_train)
        #perform each classifier on the splited balance train and test data
        for j in range(len(classifiers)):
            #get the f1 score and the probability of the value being 0 or 1
            score,probas_=classifiers[j](tr_bal,credit_bal,X_test,y_test)
            f1_score_arr[j][i]=score
        i+=1
        print(f1_score_arr)
    #caculate the average value of the f1 score for each classifier    
    for i in range(clf_num):
        cv_result[i] = f1_score_arr[i].mean(axis=0)
    print("result: " + str(cv_result))


# In[5]:


def get_f1_score(te_credit,predict):
    f1_scores=f1_score(te_credit,predict,labels=[0,1],average=None)
    return f1_scores


# In[6]:


def decision_tree(tr_bal,credit_bal,X_test,y_test):
    classifier = tree.DecisionTreeClassifier()
    f1_score =0.0
    clf_tree= BaggingClassifier(base_estimator=classifier,n_estimators=200,random_state=0)
    predict=clf_tree.fit(tr_bal, credit_bal).predict(X_test) 
    probas_=clf_tree.predict_proba(X_test)[:,1]
#    fw=open('result_dt.txt','w')
#    for i in range(len(predict)):
#        fw.write(str(predict[i]))
#        fw.write('\n')
#    fw.close()
    f1_score = get_f1_score(y_test,predict)
    return f1_score,probas_


# In[7]:


def SVM(tr_bal,credit_bal,X_test,y_test):
    f1_score =0.0
    train_sc = preprocessing.scale(tr_bal)
    test_sc = preprocessing.scale(X_test)
    svm=SVC(kernel='linear',probability=True)
    svm.fit(train_sc,credit_bal)
    predict=svm.predict(test_sc)
    probas_=svm.predict_proba(test_sc)[:,1]
    print(len(predict))
#     fw=open('result_svm.txt','w')
#     for i in range(len(predict)):
#         fw.write(str(predict[i]))
#         fw.write('\n')
#     fw.close()
    f1_score=get_f1_score(y_test,predict)
    return f1_score,probas_


# In[8]:


def naive_bayes(tr_bal,credit_bal,X_test,y_test):
    f1_score =0.0
    gnb=GaussianNB()
    gnb.fit(tr_bal,credit_bal)
    predict=gnb.predict(X_test)
    probas_=gnb.predict_proba(X_test)[:,1]
#     fw=open('result_nb.txt','w')
#     for i in range(len(predict)):
#         fw.write(str(predict[i]))
#         fw.write('\n')
#     fw.close()
    f1_score=get_f1_score(y_test,predict)
    return f1_score,probas_


# In[9]:


def adaboost(tr_bal,credit_bal,X_test,y_test):
    f1_score=0.0
    #set adaboost parameter search range for RandomizedSearchCV
    param_dist = {
        'criterion': ['gini', 'entropy'],
        'min_samples_split': randint(2, 50),
                  'max_depth': randint(1, 20),
        'min_samples_leaf': randint(2, 15),
        'max_leaf_nodes': randint(2,50)
    }
    ada_clf=tree.DecisionTreeClassifier()
    #ada_clf -- training model
    #param_distributions -- dictionary with parameters names (str) as keys and distributions or lists of parameters to try
    #cv-- determines the cross-validation splitting strategy
    #n_iter -- number of parameter settings that are sampled
    rscv = RandomizedSearchCV(ada_clf, param_distributions=param_dist, cv=10,scoring='f1',n_iter=20)
    rscv.fit(tr_bal, credit_bal)
    best_clf = rscv.best_estimator_
    adaboost = AdaBoostClassifier(best_clf,n_estimators=200,random_state=42)
    adaboost.fit(tr_bal, credit_bal)
    predict = adaboost.predict(X_test)
    probas_=adaboost.predict_proba(X_test)[:,1]
#     fw=open('result_ada.txt','w')
#     for i in range(len(predict)):
#         fw.write(str(predict[i]))
#         fw.write('\n')
#     fw.close()
    f1_score=get_f1_score(y_test,predict)
    return f1_score,probas_


# In[10]:


def random_forest(tr_bal,credit_bal,X_test,y_test):
    f1_score=0.0
#     x_train, x_test, y_train, y_test = train_test_split(tr_bal, credit_bal, random_state=0)

#     rfc = RandomForestClassifier(random_state=42, class_weight = 'balanced')
#     param_grid = { 
#         'n_estimators': [200, 500],
#         'max_features': ['auto', 'sqrt', 'log2'],
#         'max_depth' : [4,5,6,7,8],
#         'criterion' :['gini', 'entropy']
#     }
#     k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
#     CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= k_fold, scoring = 'f1')
#     CV_rfc.fit(x_train, y_train)
#     print(CV_rfc.best_params_)
#     print(CV_rfc.best_score_)
#     print(CV_rfc.best_estimator_)
    rf=RandomForestClassifier(criterion='gini',max_depth = 8,n_estimators=200)
    rf.fit(tr_bal,credit_bal)
    predict=rf.predict(X_test)
    probas_=rf.predict_proba(X_test)[:,1]
#     fw=open('result_rf.txt','w')
#     for i in range(len(predict)):
#         fw.write(str(predict[i]))
#         fw.write('\n')
#     fw.close()
    f1_score=get_f1_score(y_test,predict)
    return f1_score,probas_


# In[11]:


def main():
    read_data()
    cross_validation()
#     tr_bal,credit_bal=sampling(train2,credit)
#     decision_tree(tr_bal,credit_bal,test,y_test)
#     SVM(tr_bal,credit_bal,test,y_test)
#     naive_bayes(tr_bal,credit_bal,test,y_test)
#     random_forest(tr_bal,credit_bal,test,y_test)
#     adaboost(tr_bal,credit_bal,test,y_test)


# In[12]:


main()

