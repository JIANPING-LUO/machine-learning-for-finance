#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:13:44 2017

@author: JIANPING LUO
"""
import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
from best_strategy import *
from indicators import *

def compute_portvals(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), orders_file = "./order.csv", start_val = 100000):
    # this is the function the autograder will call to test your code
    #parse the data from orders.csv and save in the frame orders.
    orders = pd.read_csv(orders_file, index_col ='Date', 
             parse_dates = True, 
             na_values = ['nan'])
    #sort the dates
    orders = orders.sort_index()
    #get symbols from orders
    syms = orders['Symbol'].unique().tolist()

    dates = pd.date_range(sd, ed)    
    prices_all = get_data(syms, dates)  # automatically adds SPY    
    prices = prices_all[syms]  # only portfolio symbols
    #prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    prices['Cash'] = 1.0

    #create a dataframe trades to calculate the trading information
    #set the initial values for different stock to 0
    trades = prices.copy()
    for sym in syms:
        trades[sym] = 0
    trades['Cash'] = 0
    buyline = []
    sellline= []

    for i in range(orders.shape[0]):
        date = orders.index[i]
        sym  = orders.ix[i]['Symbol']
        if orders.ix[i,'Order'] == 'BUY':
            buyline.append(date)
            trades.ix[date,sym] += orders.ix[i,'Shares']
            trades.ix[date,'Cash'] -= prices.ix[date,sym] * orders.ix[i,'Shares']
            #print(prices.ix[date,sym])
        elif orders.ix[i,'Order'] == 'SELL':
            sellline.append(date)
            trades.ix[date,sym] -= orders.ix[i,'Shares']
            trades.ix[date,'Cash'] += prices.ix[date,sym] * orders.ix[i,'Shares']

    #create a dataframe holdings to calculate the holdings for each day
    holdings = trades.copy()
    holdings.ix[0,'Cash'] += start_val

    for j in range(1,holdings.shape[0]):
        s = 0
        t = 0
        leverage = 0 
        for sym in syms:
            holdings.ix[j,sym] += holdings.ix[j-1,sym]
            s += abs(holdings.ix[j,sym] * prices.ix[j,sym])
            t += holdings.ix[j,sym] * prices.ix[j,sym]
        holdings.ix[j,'Cash'] = holdings.ix[j,'Cash'] + holdings.ix[j-1,'Cash']

    #create a datafram values to calculate the values for all holdings
    values = prices * holdings

    values['portfolio'] = values.sum(axis = 1)

    portvals = values[['portfolio']]
    
    return portvals#, buyline,sellline


def compute_portfolio_stats(prices,rfr = 0.0, sf = 252.0,sv=1):
    #cr: Cumulative return
    #adr: Average daily return
    #sddr: Standard deviation of daily return
    #sr: Sharpe Ratio
    normed_price = prices / prices.ix[0,:]
    alloced_price = normed_price
    pos_vals = alloced_price * sv
    port_val = pos_vals#.sum(axis=1)
    #addition ended on 1/29/2017

    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns[1:]
    cr   = (port_val[-1] / port_val[0]) - 1.0
    adr  = daily_returns.mean()
    sddr = daily_returns.std()
    sr   = np.sqrt(sf) * (adr - rfr) / sddr
    return cr,adr,sddr,sr

def plot_svl(outputs, labels, file_name, title = 'AAPL with Decision Tree'):    
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title = title)
    ax.plot(outputs[0],label = labels[0],color = 'green')
    ax.plot(outputs[1],label = labels[1], color = 'blue')
    ax.plot(outputs[2],label = labels[2], color = 'black')
    ax.plot(outputs[3], label = labels[3], color = 'red')
    ax.plot(outputs[4], label = labels[4], color = 'yellow')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.legend(loc = 'upper center',bbox_to_anchor = (0.55,1.0), prop = {'size':7})
    plt.xticks(rotation=30)
    plt.savefig(file_name)

def plot_svm(outputs, labels, file_name, title = 'AAPL with Decision Tree'):    
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title = title)
    ax.plot(outputs[0],label = labels[0],color = 'green')
    ax.plot(outputs[1],label = labels[1], color = 'blue')
    ax.plot(outputs[2],label = labels[2], color = 'black')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.legend(loc = 'upper center',bbox_to_anchor = (0.55,1.0), prop = {'size':7})
    plt.xticks(rotation=30)
    plt.savefig(file_name)

def test(orderfiles = ['./order_out.csv'],sv = 100000,sd = dt.datetime(2010,1,1), ed = dt.datetime(2011,12,31)):
    
    outputs = []
    performance = []
    
    for orderfile in orderfiles:
        output = compute_portvals(orders_file = orderfile, sd = sd, ed = ed)
        output = output[output.columns[0]]
        output_n = normalization_indicator(output)
        outputs.append(output_n)
        cr,adr,sddr,sr = compute_portfolio_stats(output)
        performance.append((cr,adr,sddr,sr))
    return outputs, performance

def test_decision_tree():
    outputs, performance = test(orderfiles = ['decision_tree_10.csv',
                                 'decision_tree_20.csv',
                                 'decision_tree_30.csv',
                                 'decision_tree_40.csv',
                                 'decision_tree_50.csv'])
    labels = ['leaf_size_10',
              'leaf_size_20',
              'leaf_size_30',
              'leaf_size_40',
              'leaf_size_50']
    print performance
    plot_svl(outputs,labels, file_name = 'decision_tree.png', title = 'AAPL with Decision Tree')

def test_boosting():
    outputs, performance = test(orderfiles = ['boost_10.csv',
                                 'boost_20.csv',
                                 'boost_30.csv',
                                 'boost_40.csv',
                                 'boost_50.csv'])
    labels = ['bag_size_10',
              'bag_size_20',
              'bag_size_30',
              'bag_size_40',
              'bag_size_50']
    print performance
    plot_svl(outputs,labels, file_name = 'boosting.png', title = 'AAPL with Boosting')

def test_knn():
    outputs, performance = test(orderfiles = ['knn_10.csv',
                                 'knn_20.csv',
                                 'knn_30.csv',
                                 'knn_40.csv',
                                 'knn_50.csv'])
    labels = ['k_value_10',
              'k_value_20',
              'k_value_30',
              'k_value_40',
              'k_value_50']
    print performance
    plot_svl(outputs,labels, file_name = 'knn.png', title = 'AAPL with KNN')

def test_svm():
    outputs, performance = test(orderfiles = ['svm_linear.csv',
                                 'svm_poly.csv',
                                 'svm_rbf.csv'])
    labels = ['linear',
              'poly',
              'rbf']
    print performance
    plot_svm(outputs,labels, file_name = 'svm.png', title = 'AAPL with SVM')

def test_nnl():
    outputs, performance = test(orderfiles = ['nnl_1.csv',
                                 'nnl_2.csv',
                                 'nnl_3.csv',
                                 'nnl_4.csv',
                                 'nnl_5.csv'])
    labels = ['1_hidden_layer',
              '2_hidden_layer',
              '3_hidden_layer',
              '4_hidden_layer',
              '5_hidden_layer']
    print performance
    plot_svl(outputs,labels, file_name = 'nnl.png', title = 'AAPL with Neural Network')

if __name__ == "__main__":
    test_decision_tree()
    test_boosting()
    test_knn()
    test_svm()
    test_nnl()
