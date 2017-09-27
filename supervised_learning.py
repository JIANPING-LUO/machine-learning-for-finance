import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import datetime as dt
import os
import math
from util import get_data, plot_data
from indicators import *
from best_strategy import build_benchmark
import csv
import RTLearner as rtl
import BagLearner as bl
import BoostLearner as bol
import KNNLearner as knn
import SVMLearner as svm
import NeuralNetworkLearner as nnl



### Use indicators to make some kind of trading decision for each day
def build_RL_data(sd = dt.datetime(2008,1,1),ed = dt.datetime(2009,12,31), syms = ['AAPL'], lookback = 21):
        
        
    price, price_all,price_SPY = compute_prices(sd,ed)
        
    sma = price_sma_indicator(price)
    bbp,top_band,bottom_band = bollinger_band_indicator(price)
    rsi = rsi_indicator(price)
    stoch,stochd = stochastic_indicator(price)
        
    # Orders starts as a NaN array of the same shape/index as price
    Ydata = price.copy()
    Ydata.values[:-lookback,:] = price.values[lookback:,:] / price.values[:-lookback,:] - 1
    Ydata.values[-lookback:,:] = np.NaN
    Ydata = Ydata.fillna(method = 'ffill')
    
    YBUY = 0.01
    YSELL = -0.01
    
    for i in range(Ydata.shape[0]):
        if Ydata.values[i,:] > YBUY:
            Ydata.values[i,:] = 1.0
        elif Ydata.values[i,:] < YSELL:
            Ydata.values[i,:] = -1.0
        else:
            Ydata.values[i,:] = 0.0
    
    sma_sd = standardization_indicator(sma)
    bbp_sd = standardization_indicator(bbp)
    rsi_sd = standardization_indicator(rsi)
    stoch_sd = standardization_indicator(stoch)
    
    sma_sd = sma_sd.rename(columns = {'AAPL':'sma'})
    bbp_sd = bbp_sd.rename(columns = {'AAPL':'bbp'})
    rsi_sd = rsi_sd.rename(columns = {'AAPL':'rsi'})
    stoch_sd = stoch_sd.rename(columns = {'AAPL':'stoch'})
    Ydata = Ydata.rename(columns = {'AAPL':'Ydata'})
    
    train_data = pd.concat([sma_sd,bbp_sd,rsi_sd,stoch_sd,Ydata], axis = 1)
    
    return train_data

def training_data(train_data,test_data,learner):
    
    
    trainX = train_data.values[:, 0:-1]
    trainY = train_data.values[:, -1]
    testX = test_data.values[:, 0:-1]
    testY = test_data.values[:, -1]
    
    learner.addEvidence(trainX, trainY)  # train it

    # evaluate in sample
    train_predY = learner.query(trainX)  # get the predictions
    rmse_train = math.sqrt(((trainY - train_predY) ** 2).sum()/trainY.shape[0])

    print "In sample results"
    print "RMSE: ", rmse_train
    c_train = np.corrcoef(train_predY, y=trainY)
    print "corr: ", c_train[0, 1]

    # evaluate out of sample
    test_predY = learner.query(testX)  # get the predictions
    rmse_test = math.sqrt(((testY - test_predY) ** 2).sum()/testY.shape[0])

    print "Out of sample results"
    print "RMSE: ", rmse_test
    c_test = np.corrcoef(test_predY, y=testY)
    print "corr: ", c_test[0, 1]
    performance = rmse_train, c_train, rmse_test, c_test
    return train_predY,test_predY, performance

def build_orders(learner, syms = ['AAPL'], lookback = 21, filename = 'order_decision_tree.csv'):
        
    train_data = build_RL_data()
    test_data = build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    train_predY, test_predY, performance = training_data(train_data,test_data, learner)
    
    price, price_all,price_SPY = compute_prices(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31))
    
    price_out,price_all_out,price_SPY_out = compute_prices(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    
    orders = price_out.copy()
    orders.ix[:,:] = 0

    orders[(test_predY > 0.3)] = 1
    orders[(test_predY < -0.3)] = -1
    
    for i in range(orders.shape[0]):
        if orders.values[i] == 1 or orders.values[i] == -1 :
            orders.ix[i+1:i+lookback,:] = 0
    
    for i in range(orders.shape[0]):
        if orders.values[i] > 2:
            continue
        elif orders.values[i] < -2:
            continue
        elif orders.values[i] > 0:
            if i+lookback < orders.shape[0]:
                if orders.values[i+lookback] > 2:
                    continue
                elif orders.values[i+lookback] < -2:
                    continue
                elif orders.values[i+lookback] == 0:
                    orders.values[i+lookback] = -3
                else:
                    orders.values[i+lookback] -= 1
        elif orders.values[i] < 0:
            if i+lookback < orders.shape[0]:
                if orders.values[i+lookback] > 2:
                    continue
                elif orders.values[i+lookback] < -2:
                    continue
                elif orders.values[i+lookback] == 0:
                    orders.values[i+lookback] = 3
                else:
                    orders.values[i+lookback] += 1
    
    #print orders

    ###And more importantly, drop all rows with no non-zero values(i.e. no orders)
    orders = orders.loc[(orders != 0).any(axis = 1)]
        
    ###Now we have only the days that have orders. That's better, at least!
    order_list = []
    for day in orders.index:
        for sym in syms:
            if orders.ix[day,sym] > 2:
                order_list.append([day.date(), sym, 'BUY', 200])
            elif orders.ix[day,sym] > 1:
                order_list.append([day.date(), sym, 'BUY', 400])
            elif orders.ix[day,sym] > 0:
                order_list.append([day.date(), sym, 'BUY', 200])
            elif orders.ix[day,sym] < -2:
                order_list.append([day.date(), sym, 'SELL', 200])
            elif orders.ix[day,sym] < -1:
                order_list.append([day.date(), sym, 'SELL', 400])
            elif orders.ix[day,sym] < 0:
                order_list.append([day.date(), sym, 'SELL', 200])
        
        
    with open(filename,'wb') as csvfile:
        fieldnames = ['Date','Symbol','Order','Shares']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        for order in order_list:
            order_tmp = ",".join(str(x) for x in order)
            csvfile.write(order_tmp + "\n")
def test_decision_tree():
    train_data = build_RL_data()
    test_data = build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    
    build_orders(learner = rtl.RTLearner(leaf_size = 10), filename = 'decision_tree_10.csv')
    build_orders(learner = rtl.RTLearner(leaf_size = 20), filename = 'decision_tree_20.csv')
    build_orders(learner = rtl.RTLearner(leaf_size = 30), filename = 'decision_tree_30.csv')
    build_orders(learner = rtl.RTLearner(leaf_size = 40), filename = 'decision_tree_40.csv')
    build_orders(learner = rtl.RTLearner(leaf_size = 50), filename = 'decision_tree_50.csv')
    
def test_boosting():
    train_data = build_RL_data()
    test_data = build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    
    build_orders(learner = bol.BoostLearner(learner=rtl.RTLearner, kwargs={'k': 20}, bags=10, verbose=False), filename = 'boost_10.csv')
    build_orders(learner = bol.BoostLearner(learner=rtl.RTLearner, kwargs={'k': 20}, bags=20, verbose=False), filename = 'boost_20.csv')
    build_orders(learner = bol.BoostLearner(learner=rtl.RTLearner, kwargs={'k': 20}, bags=30, verbose=False), filename = 'boost_30.csv')
    build_orders(learner = bol.BoostLearner(learner=rtl.RTLearner, kwargs={'k': 20}, bags=40, verbose=False), filename = 'boost_40.csv')
    build_orders(learner = bol.BoostLearner(learner=rtl.RTLearner, kwargs={'k': 20}, bags=50, verbose=False), filename = 'boost_50.csv')

def test_knn():
    train_data = build_RL_data()
    test_data = build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    
    build_orders(learner = knn.KNNLearner(k = 10), filename = 'knn_10.csv')
    build_orders(learner = knn.KNNLearner(k = 20), filename = 'knn_20.csv')
    build_orders(learner = knn.KNNLearner(k = 30), filename = 'knn_30.csv')
    build_orders(learner = knn.KNNLearner(k = 40), filename = 'knn_40.csv')
    build_orders(learner = knn.KNNLearner(k = 50), filename = 'knn_50.csv')
      
def test_svm():
    train_data = build_RL_data()
    test_data = build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    
    build_orders(learner = svm.SVMLearner(kernel = 'linear'), filename = 'svm_linear.csv')
    build_orders(learner = svm.SVMLearner(kernel = 'poly'), filename = 'svm_poly.csv')
    build_orders(learner = svm.SVMLearner(kernel = 'rbf'), filename = 'svm_rbf.csv')

def test_nnl():
    train_data = build_RL_data()
    test_data = build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    
    build_orders(learner = nnl.NeuralNetworkLearner((30)), filename = 'nnl_1.csv') 
    build_orders(learner = nnl.NeuralNetworkLearner((30,30)), filename = 'nnl_2.csv') 
    build_orders(learner = nnl.NeuralNetworkLearner((30,30,30)), filename = 'nnl_3.csv') 
    build_orders(learner = nnl.NeuralNetworkLearner((30,30,30,30)), filename = 'nnl_4.csv') 
    build_orders(learner = nnl.NeuralNetworkLearner((30,30,30,30,30)), filename = 'nnl_5.csv') 

def test_code():
    train_data = build_RL_data()
    test_data = build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    training_data(train_data,test_data)
    build_RL_orders()

if __name__ =="__main__":
    
    test_decision_tree()
    test_boosting()
    test_knn()
    test_svm()
    test_nnl()
    print 'good job'