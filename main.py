#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:04:18 2018

@author: yueningli
"""
import sys
sys.path.append('../Wide&Deep')
from readindi import readindividual
from iForest import iforest
from preprocessing import preprocessing
from preprocessing import add_flag
from DeepnWide import deep_wide_classify
import pandas as pd
import time

columns_list = ['STATUS', 'OPERATING_SYSTEM', 'DEVICE_TYPE', 'BROWSER', 'client_connectionType',
 'APPLICATION_NAME', 'client_country', 'client_city', 'client_isp', 'client_organization']

def CatForest(filename,threshold):
    data, topnkey=readindividual(filename)
    #print(data[:10])
    print('Topnkey:',len(topnkey))
    _, _, p  = preprocessing(data, topnkey, 1)
    df = p[:2]
    df = df.drop(df.index[:2])
    total_num_examples = 0
    for num in range(1,len(topnkey)):
    #for num in range(1,3):
        category,user, predict =preprocessing(data,topnkey,num)
        total_num_examples += len(user)
        u = iforest(category,user,threshold)
        new_data  = predict.iloc[u,:]
        print('Anomalies detected:',len(new_data))
        #print(new_data)
        df = pd.concat([df, new_data], ignore_index=True)
        #print(df.shape)
    df = add_flag(df)
    #print(df.shape)
    print('Total number of examples:', total_num_examples)
    return df

def __main():
    time1 = time.time()
    filename='login_20180109.csv'
    time2 = time.time()
    print('Time for reading file:', time2-time1)
    #threshold = -0.28
    threshold = -0.15
    dataframe = CatForest(filename,threshold)
    time3 = time.time()
    print('Time for Isolation Forest:', time3-time2)

    length = len(dataframe)
    train_idx = (length * 8)//10
    print('Number of anomalies passed to DNN: %d' %(length))
    print('Number of training examples: %d test examples: %d'%(train_idx, length - train_idx))
    train_data = dataframe[:train_idx]
    test_data = dataframe[train_idx:]
    deep_wide_classify(train_data, test_data)


__main()
