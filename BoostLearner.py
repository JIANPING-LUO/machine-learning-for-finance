#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:13:44 2017

@author: JIANPING LUO
"""
import numpy as np
from random import *


class BoostLearner(object):
    
    def __init__(self, learner , kwargs = {"k":3}, bags = 20 , boost = False, verbose = False):
        self.learner = learner;
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        
    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        
        learners = [] #empty list that will carry our learners
        XtrainSize = self.Xtrain.shape[0]
        num = 0 
        bagsize = int(np.floor(XtrainSize*1))
      
        #intialize first bag
        newBag = np.random.randint(0,XtrainSize,bagsize)
        while num < self.bags:
            bagSetTrainX = self.Xtrain[newBag,:]
            bagSetTrainY = self.Ytrain[newBag]
            learner = self.learner(self.kwargs['k'], verbose = False) # contruct new learner
            learner.addEvidence(bagSetTrainX, bagSetTrainY) # train our leanrer with training set
            learners.append(learner) #add leanrer to the list 
            res = np.zeros((XtrainSize,len(learners)))
            a = 0
            # for each learner, gather predictions in a 2D array
            for lea in learners:
                res[:,a]= lea.query(self.Xtrain)
                a += 1
            Response = np.mean(res, axis=1) #average the predictions of the learners added
            err = np.abs(Response-self.Ytrain)           
            #create new bag using weight found with the previous lerners
            newBag = np.random.choice(np.arange(self.Xtrain.shape[0]),XtrainSize,p = err/err.sum())
            num += 1
        
        self.learners = learners
    def query(self,points):
        
        Ypredict = None
        for learner in self.learners:
            
            R = learner.query(points)
            if Ypredict is None:
                Ypredict = R
            else:
                Ypredict = np.add(Ypredict, R)
        
        Ypredict = Ypredict / len(self.learners)
        return Ypredict

if __name__=="__main__":
    print "nice job"
    
    