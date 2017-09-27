#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:13:44 2017

@author: JIANPING LUO
"""
import numpy as np
import sys, csv, math


class KNNLearner(object):

    def __init__(self,k, verbose = False):
        self.k = k

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.dataX = dataX
        self.dataY = dataY

    def getknearpos(self,k,array,value):
        nn = np.linalg.norm(array - value,axis=1).argsort()
        nn = nn[0:k]
        
        return nn    

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        
        res = np.zeros(points.shape[0])
        o=0
        for a in points:
            neighbors = np.empty
            neighbors = self.getknearpos(self.k,self.dataX,a)
            av = self.dataY[neighbors].mean()
            res[o]= av
            o +=1
      
        return res    

if __name__=="__main__":
    print "Nice Job"