#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:13:44 2017

@author: JIANPING LUO
"""
import numpy as np
import sys, csv, math
from sklearn.neural_network import MLPClassifier

class NeuralNetworkLearner(object):

    def __init__(self,hidden_layer_sizes = (5,2), verbose = False):
        self.hidden_layer_sizes = hidden_layer_sizes

    def addEvidence(self,dataX,dataY):
        
        self.dataX = dataX
        self.dataY = dataY
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = self.hidden_layer_sizes, random_state=1)
        self.clf.fit(self.dataX,self.dataY)
       
    def query(self,testX):
        
        return self.clf.predict(testX)  

if __name__=="__main__":
    print "Nice Job"