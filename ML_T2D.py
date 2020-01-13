#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:44:14 2020

@author: kaborekader
"""

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import seaborn as sns

# Binary Classification with Sonar Dataset: Standardized Larger
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn as sk

from sklearn.linear_model import LassoCV,RidgeCV,ElasticNetCV
from sklearn.feature_selection import SelectFromModel
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions
from mlxtend.classifier import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier



from keras.models import Sequential
from keras.layers import Dense
# load dataset

# 0 healthy 1 diabet
data_diabet = read_csv("t2d.csv", header=None)

data_healthy = read_csv("healthy.csv", header=None)

ones = np.ones(len(data_diabet))
zeros = np.zeros(len(data_healthy))
data_diabet[1456] = ones
data_healthy[1456] = zeros


frames = [data_diabet, data_healthy]
result = pd.concat(frames)

dataframe = sk.utils.shuffle(result)


dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:-1].astype(float)
Y = dataset[:,-1]
y = Y.astype(int)





from sklearn.neural_network import MLPClassifier
from sklearn import svm


from mlxtend.classifier import EnsembleVoteClassifier






clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=2000, max_depth=200, random_state=1)
clf3 = GaussianNB()
clf4 =  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1)
clf5 =  KNeighborsClassifier(n_neighbors=4)
clf6 =  svm.SVC(decision_function_shape="ovo")
clf7 =  LogitBoost(n_estimators=200, random_state=0)



clf8 = Pipeline([
  ('feature_selection', SelectFromModel(RidgeCV())),
  ('classification', svm.SVC(decision_function_shape="ovo"))
])
    
clf8.fit(X, y)



clf9 = Pipeline([
  ('feature_selection', SelectFromModel(ElasticNetCV())),
  ('classification', svm.SVC(decision_function_shape="ovo"))
])
    
clf9.fit(X, y)


clf10 = Pipeline([
  ('feature_selection', SelectFromModel(LassoCV())),
  ('classification', svm.SVC(decision_function_shape="ovo"))
])
    
clf9.fit(X, y)


eclf = EnsembleVoteClassifier(clfs=[ clf1, clf6], weights=[1,1])
clf = clf10
print('5-fold cross validation:\n')

labels = ['ridge', 'elstic net','lasso']

#clf1, clf2, clf3,clf4,clf5,clf6,clf7, eclf
labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes','MLP',"KNN","SVM","LogiBoost", 'Ensemble']
for clf, label in zip([clf1, clf2, clf3,clf4,clf5,clf6,clf7, eclf], labels):

    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=10, 
                                              scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
    
    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=10, 
                                              scoring='f1')
    print("F1: %0.4f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))


    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=10, 
                                              scoring='recall')
    print("recall: %0.4f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))


    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=10, 
                                              scoring='precision')
    print("precision: %0.4f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
    
    
    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=10, 
                                              scoring='roc_auc')
    print("roc_auc: %0.4f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label)) 
    
    
    
   
    
    
    





    # create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)




#visualization
plt.figure(figsize=(12,10))
cor = dataframe.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

ax = sns.heatmap(dataframe.corr(), cmap="YlGnBu")
ax = sns.heatmap(dataframe, linewidths=.5)

ax = sns.heatmap(dataframe, annot=True, fmt="d")







# NEURAL NETWORKS
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# larger model
def create_larger():
	# create model
	model = Sequential()
	model.add(Dense(1455, input_dim=1455, activation='relu'))
	model.add(Dense(30, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=30, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))