# -*- coding: utf-8 -*-

import json
import sys
import math

#from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing

#english_stopwords = stopwords.words('english')
#english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']


IS_LOCAL = True

def load():
    TRAIN_N = 0
    train_docs = []
    train_docs_topics = []

    TEST_N = 0
    test_docs = []

    if IS_LOCAL:
        source = open('C:/Users/dwoo57/Google Drive/Career/Data Mining Competitions/Quora/ML Code Sprint 2014/Labeler/Data/labeler_sample.in').readlines()
    else:
        source = sys.stdin
    for ind, line in enumerate(source):
        if ind == 0:
            tmp_var = line.split()
            TRAIN_N = int(tmp_var[0])
            TEST_N =int(tmp_var[1])
        elif 1 <= ind <= TRAIN_N*2 and ind%2 ==0:
            train_docs.append(line.strip())
        elif 1 <= ind <= TRAIN_N*2 and ind%2 ==1:
            train_docs_topics.append(line.split())
        elif ind >= TRAIN_N*2 + 1 and ind<= TRAIN_N*2+TEST_N+1:
            #TEST_N = int(line.strip())
            test_docs.append(line.strip())
        #else:
            #test_docs.append(line.strip())

    return train_docs,train_docs_topics, test_docs
    
def verify(actual_dict, pred_dict):
    correct_count = 0
    for k, v in pred_dict.items():
        true_v = actual_dict[k]
        if true_v == v:
            correct_count += 1
    return float(correct_count) / len(pred_dict)


def text_clf(train_docs,train_docs_topics, test_docs):
    #vectorizer = CountVectorizer(stop_words=english_stopwords)
    vectorizer = CountVectorizer(stop_words=english_stopwords)
    #corpus = [get_text(doc) for doc in train_docs]
    X = vectorizer.fit_transform(train_docs)
    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(X)
    
    #now get the topics for training
    #lb = preprocessing.LabelBinarizer()
    lb = preprocessing.MultiLabelBinarizer()
    Y = lb.fit_transform(train_docs_topics)
    
    classifier = Pipeline([
    #('vectorizer', CountVectorizer(min_n=1,max_n=2)),
    #('vectorizer', CountVectorizer(stop_words=english_stopwords)),
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
    #('clf', OneVsRestClassifier(MultinomialNB()))])    
    
    classifier.fit(train_docs, Y)
    #predicted = classifier.predict(train_docs)    
    predicted = classifier.predict(test_docs)
    
    all_labels = lb.inverse_transform(predicted)
    
    for item, labels in zip(test_docs, all_labels):
        print(' '.join(labels))

def build_model(train_docs,train_docs_topics, test_docs):
    text_clf(train_docs,train_docs_topics, test_docs)

# main program starts here
train_docs,train_docs_topics, test_docs = load()
build_model(train_docs,train_docs_topics,test_docs)
#text_clf(train_docs,test_docs)