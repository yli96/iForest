#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:16:46 2018

@author: yueningli
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
def iforest(category,user,threshold):
    length=len(category)
    rng = np.random.RandomState(42)
    clf = IsolationForest(max_samples=length, random_state=rng)
    clf.fit(category)
    b=clf.decision_function(category)
    print('Minimum decision value for user %d: %f' %( user['prsId'][user.index[0]], min(b)))
    
    '''
    for i in range(0,length):
        if b[i]<threshold:
            #print 'Anomaly Detected at:', i
            print(user.index[i]+2)
            print(user['prsId'][user.index[i]])
    '''
    return np.where(b<threshold)[0]