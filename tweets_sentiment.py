#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 20:45:22 2018

@author: leli
"""
import os
os.chdir('/Users/leli/iAdvize/tweets_analysis/')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import preprocessing as pre

from time import time



## import raw data

raw_file = "/Users/leli/iAdvize/data/training.1600000.processed.noemoticon_translated.csv" 

df = pd.read_csv(raw_file, sep = ',', usecols = ['target', 'translated'])

df = df.dropna(axis = 0, how = 'any')

df = df.loc[df['target'] != 2] ## remove neutral tweets (0 = negative, 2 = neutral, 4 = positive)

df.loc[df['target'] == 0, 'target'] = 0  ## negative 
df.loc[df['target'] == 4, 'target'] = 1  ## positive

df = df.sample(frac=1).reset_index(drop=True)


## when dataset is imbalanced, using oversampling method such as SMOTE.
print ('number of negative tweets:{}'.format(len(np.where(df['target'] == 0)[0])))
print ('number of positive tweets:{}'.format(len(np.where(df['target'] == 1)[0])))



## preprocessing (0 = negative, 2 = neutral, 4 = positive)




X = np.array(df['translated'].apply(lambda x: pre.preprocess(x)).tolist())
Y = np.array(df['target'].tolist())



np.random.seed(0) 
train_index = np.random.choice(len(df), size = int(0.8*len(df)), replace = False)
test_index = np.array(list(set(range(len(df))) - set(train_index)))

X_train = list(X[train_index])
y_train = list(Y[train_index])

X_test = list(X[test_index])
y_test = list(Y[test_index])




#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state= 11)


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(X_train,
                             max_df = 0.95,
                             min_df = 2,
                             stop_words = None,
                             tokenizer = None)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)  ## new features in X_test will be ignored when doing transform

nbr_of_features = len(vectorizer.get_feature_names())

print('Vocabulary size is {}'.format(nbr_of_features))


## define a reporting function to show top models

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



##############################################################################


## build Multinomial Naive Bayesian clf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import MultinomialNB


MNBclf = MultinomialNB()


param_dist = {'alpha': np.linspace(0.01, 1, num = 200)}


n_iter_search = 200
random_search = RandomizedSearchCV(MNBclf, 
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search, 
                                   cv = 5,      ## since cross validation is used, no need to separate training and test sample at the beginning.
                                   n_jobs = -1) ## None means 1 and -1 means all processors. The data will be copied for each processor, attention when data
                                                ## is large since it surpass the memory of system. 



start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)








## Bernoulle Naive Bayesian (remark: only work for binary data)

NBclf = BernoulliNB()
n_iter_search = 20
random_search = RandomizedSearchCV(NBclf, 
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search, 
                                   cv = 5,      ## since cross validation is used, no need to separate training and test sample at the beginning.
                                   n_jobs = -1) ## None means 1 and -1 means all processors. The data will be copied for each processor, attention when data
                                                ## is large since it surpass the memory of system. 

X_binary = X_train
X_binary[X_binary >=1] = 1


start = time()

random_search.fit(X_binary, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)



##############################################################################


## build SVM clf
from sklearn import svm

SVMclf = svm.SVC()
SVMclf.fit(X_train, y_train)

param_dist = {"C": np.linspace(0.01, 10, num = 20),
              "kernel": ['poly', 'rbf'],
              "gamma": np.linspace(0.01, 10, num = 20)}



n_iter_search = 10
random_search = RandomizedSearchCV(SVMclf, 
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search, 
                                   cv = 5,      ## since cross validation is used, no need to separate training and test sample at the beginning.
                                   n_jobs = -1) ## None means 1 and -1 means all processors. The data will be copied for each processor, attention when data
                                                ## is large since it surpass the memory of system. 



start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)





##############################################################################



# build RF classifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint


N = 20  ## number of random trees 

clf = RandomForestClassifier(n_estimators = N)



param_dist = {"max_depth": np.linspace(3,10, num = 8),
              "max_features": sp_randint(2, max(int(nbr_of_features/100), 2)),  ## at least two features are considered
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 200 ## select from 20 combinations of parameters.
random_search = RandomizedSearchCV(clf, 
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search, 
                                   cv = 5,      ## since cross validation is used, no need to separate training and test sample at the beginning.
                                   n_jobs = -1) ## None means 1 and -1 means all processors. The data will be copied for each processor, attention when data
                                                ## is large since it surpass the memory of system. 

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)




##############################################################################
##############################################################################
##############################################################################
## tf-idf vectorization
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer(smooth_idf=False)
X_train_tfidf = tfidf_transformer.fit_transform(X_train)


## build SVM clf

SVMclf = svm.SVC()

param_dist = {"C": np.linspace(0.01, 10, num = 20),
              "kernel": ['poly', 'rbf'],
              "gamma": np.linspace(0.01, 10, num = 20)}



n_iter_search = 10
random_search = RandomizedSearchCV(SVMclf, 
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search, 
                                   cv = 5,      ## since cross validation is used, no need to separate training and test sample at the beginning.
                                   n_jobs = -1) ## None means 1 and -1 means all processors. The data will be copied for each processor, attention when data
                                                ## is large since it surpass the memory of system. 



start = time()
random_search.fit(X_train_tfidf, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)




# build RF classifier
N = 20  ## number of random trees 

clf = RandomForestClassifier(n_estimators = N)



param_dist = {"max_depth": np.linspace(3,10, spacing = 8),
              "max_features": sp_randint(sp_randint(2, max(int(nbr_of_features/100), 2))),  ## at least two features are considered
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 200 ## select from 200 combinations of parameters.
random_search = RandomizedSearchCV(clf, 
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search, 
                                   cv = 5,      ## since cross validation is used, no need to separate training and test sample at the beginning.
                                   n_jobs = -1) ## None means 1 and -1 means all processors. The data will be copied for each processor, attention when data
                                                ## is large since it surpass the memory of system. 

start = time()
random_search.fit(X_train_tfidf, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)






######################################################################################################
######################################################################################################

## neural embedding 

X_dense_train = []
with open("/Users/leli/iAdvize/tweets_analysis/tweet_training_dense_embedding.txt", 'r') as f:
    for line in f.readlines():
        X_dense_train.append([float(x) for x in line.split(' ')[-301:-1]])
        
X_dense_train = np.array(X_dense_train)        
      

## SVM

from sklearn import svm

SVMclf = svm.SVC()
SVMclf.fit(X_train, y_train)

param_dist = {"C": np.linspace(0.01, 10, num = 20),
              "kernel": ['poly', 'rbf'],
              "gamma": np.linspace(0.01, 10, num = 20)}



n_iter_search = 10
random_search = RandomizedSearchCV(SVMclf, 
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search, 
                                   cv = 5,      ## since cross validation is used, no need to separate training and test sample at the beginning.
                                   n_jobs = -1) ## None means 1 and -1 means all processors. The data will be copied for each processor, attention when data
                                                ## is large since it surpass the memory of system. 



start = time()
random_search.fit(X_dense_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)








  
## RF
N = 20  ## number of random trees 

clf = RandomForestClassifier(n_estimators = N)



param_dist = {"max_depth": np.linspace(3,10, num = 8),
              "max_features": sp_randint(2, max(int(nbr_of_features/100), 2)),  ## at least two features are considered
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 200 ## select from 200 combinations of parameters.
random_search = RandomizedSearchCV(clf, 
                                   param_distributions = param_dist,
                                   n_iter = n_iter_search, 
                                   cv = 5,      ## since cross validation is used, no need to separate training and test sample at the beginning.
                                   n_jobs = -1) ## None means 1 and -1 means all processors. The data will be copied for each processor, attention when data
                                                ## is large since it surpass the memory of system. 

start = time()
random_search.fit(X_dense_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


       



























with open('tweet_supervised_training.txt', 'w') as f:
    
    for text, lab in zip(X[train_index], Y[train_index]):
        if lab == 0:
            f.write(text+' ' + '__label__' + 'negative\n')
        if lab == 1:
            f.write(text+' ' + '__label__' + 'positive\n')
        else:
            pass
            
    
with open('tweet_supervised_test.txt', 'w') as f:
    
    for text, lab in zip(X[test_index], Y[test_index]):
        if lab == 0:
            f.write(text+' ' + '__label__' + 'negative\n')
        if lab == 1:
            f.write(text+' ' + '__label__' + 'positive\n')
        else:
            pass
            
    
import fasttext

classifier = fasttext.supervised('tweet_supervised_training.txt', "tweet_classification_model", label_prefix='__label__')

result = classifier.test('tweet_supervised_test.txt')
print ('P@1:', result.precision)
print ('R@1:', result.recall)
print ('Number of examples:', result.nexamples)










 


