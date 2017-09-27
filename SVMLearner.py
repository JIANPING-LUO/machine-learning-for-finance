#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:13:44 2017

@author: JIANPING LUO
"""
import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import svm


class SVMLearner(object):
    
    def __init__(self, kernel = 'rbf', c = 1, gamma = 'auto'):
        self.kernel = kernel
        self.C = c
        self.gamma = gamma
    
    def addEvidence(self, X, y):
        self.model = svm.SVC(kernel = self.kernel, C = self.C, gamma = self.gamma)
        self.model.fit(X,y)
        self.model.score(X,y)
    
    def query(self,X):
        predicted = self.model.predict(X)
        return predicted

     
if __name__=="__main__":
    print "nice job"