#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:47:52 2018

@author: yueningli
"""
import pandas as pd
import numpy as np
def readindividual(filename):
    data = pd.read_csv(filename)
    idCount={}
    for id in data['prsId']:
        idCount[id]=idCount.get(id,0.0)+1.0
    #sort dictionary
    idkey = sorted(idCount.items(), key=lambda d:d[1], reverse = True)
    #top n
    #you could define n's value whatever you want only if n is less than len(idCount)
    n=len(idCount)//2
    topntemp=np.array(idkey[:n])
    topnkey=topntemp[:,0]
    #top n contains:
    totalnum=sum(topntemp[:,1])-topntemp[0][1]
    return data, topnkey