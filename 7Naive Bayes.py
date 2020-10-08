# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:06:08 2020

@author: tom05
"""


### Example – filtering mobile phone spam with the Naive Bayes algorithm

import pandas as pd
#Import mobile text message data set
sms_raw = pd.read_csv("./_data/sms_spam.csv")
# type：Spam or normal text ，text：SMS text content
print(sms_raw.dtypes)

#type frequency distribution, ham is the majority,But not excessively unbalanced
print(sms_raw['type'].value_counts()/len(sms_raw['type']))

# Python natural language processing toolset(Natural Language ToolKit)
import nltk

token_list0 = [nltk.word_tokenize(txt) for txt in
sms_raw['text']]
print(token_list0[3][1:7])

token_list1 = [[word.lower() for word in doc]
for doc in token_list0]
print(token_list1[3][1:7])

from nltk.corpus import stopwords
# 153 English stop words
print(len(stopwords.words('english')))

#The stop word or has been removed
token_list2 = [[word for word in doc if word not in
stopwords.words('english')] for doc in token_list1]
print(token_list2[3][1:7])

import string
token_list3 = [[word for word in doc if word not in
string.punctuation] for doc in token_list2]
print(token_list3[3][1:7])

# String derivation removes all Numbers , '4' is missing
token_list4 = [[word for word in doc if not word.isdigit()]
for doc in token_list3]
print(token_list4[3][1:7])

token_list5 = [[''.join([i for i in word if not i.isdigit()
and i not in string.punctuation]) for word in doc]
for doc in token_list4]

print(token_list5[3][1:7])

# String derivation removes empty elements
token_list6 =[list(filter(None, doc)) for doc in token_list5]
print(token_list6[3][1:7])

from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

token_list6 = [[lemma.lemmatize(word) for word in doc]
for doc in token_list6]
print(token_list6[3][1:7])

token_list7 = [' '.join(doc) for doc in token_list6]
print(token_list7[:2])

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
# Declaring the empty mold
vec = CountVectorizer()

X = vec.fit_transform(token_list7)

print(type(X))

print(X[:2])

sms_dtm = pd.DataFrame(X.toarray(),
columns=vec.get_feature_names())

print(sms_dtm.shape)

print(len(vec.get_feature_names())) # There are 7,612 words.

print(vec.get_feature_names()[300:305])

print(np.argwhere(sms_dtm['app'] > 0))

# Part of DTM
print(sms_dtm.iloc[4460:4470, 300:305])

# Training and test sets are segmented (sms_raw, sms_dtm, token_list6)
sms_raw_train = sms_raw.iloc[:4170, :]
sms_raw_test = sms_raw.iloc[4170:, :]
sms_dtm_train = sms_dtm.iloc[:4170, :]
sms_dtm_test = sms_dtm.iloc[4170:, :]
token_list6_train = token_list6[:4170]
token_list6_test = token_list6[4170:]

print(sms_raw_train['type'].value_counts()/
len(sms_raw_train['type']))

print(sms_raw_test['type'].value_counts()/
len(sms_raw_test['type']))

tokens_train = [token for doc in token_list6_train
for token in doc]
print(len(tokens_train))

tokens_train_spam = [token for is_spam, doc in
zip(sms_raw_train['type'] == 'spam' , token_list6_train)
if is_spam for token in doc]

tokens_train_ham = [token for is_ham, doc in
zip(sms_raw_train['type'] == 'ham' , token_list6_train)
if is_ham for token in doc]

str_train = ','.join(tokens_train)
str_train_spam = ','.join(tokens_train_spam)
str_train_ham = ','.join(tokens_train_ham)

# Python text cloud suite
from wordcloud import WordCloud

wc_train = WordCloud(background_color="white",
prefer_horizontal=0.5)
# Incoming data statistics, and production of text cloud
wc_train.generate(str_train)

import matplotlib.pyplot as plt
plt.imshow(wc_train)
plt.axis("off")
# plt.show()
# plt.savefig('wc_train.png')

from sklearn.naive_bayes import MultinomialNB
# Model definition, adaptation and prediction
clf = MultinomialNB()

clf.fit(sms_dtm_train, sms_raw_train['type'])
train = clf.predict(sms_dtm_train)
print(" The training set accuracy rate is {}".format(sum(sms_raw_train['type'] ==
train)/len(train)))

pred = clf.predict(sms_dtm_test)
print(" The test set accuracy is {}".format(sum(sms_raw_test['type'] ==
pred)/len(pred)))

# The number of samples used for training
print(clf.class_count_)

print(clf.feature_count_)

print(clf.feature_count_.shape)

print(clf.feature_log_prob_[:, :4])

print(clf.feature_log_prob_.shape)

from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):

    cv = KFold(n_splits=K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print("{} results of cross-validation are as follows：\n{}".format(K, scores))
    tmp = " Average accuracy：{0:.3f}(+/- Standard error {1:.3f})"
    print(tmp.format(np.mean(scores), sem(scores)))
evaluate_cross_validation(clf, sms_dtm, sms_raw['type'], 5)
