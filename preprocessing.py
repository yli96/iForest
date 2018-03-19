#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:24:21 2018

@author: yueningli
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" 
Created on Fri Nov 24 10:17:43 2017

@author: yueningli
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

columns_list = ['STATUS', 'OPERATING_SYSTEM', 'DEVICE_TYPE', 'BROWSER', 'client_connectionType',
 'APPLICATION_NAME', 'client_country', 'client_city', 'client_isp', 'client_organization']

def preprocessing(data,topnkey,num):

    user =data[data['prsId']==topnkey[num]]

    length=len(user)
    col=10
    mu,sigma=0,0.00001
    predict=np.zeros((length,col))   
    category=np.zeros((length,col))
    sa=np.random.normal(mu,sigma,length)
    
    for i in range(0,length):
        if user['STATUS'][user.index[i]]!='SUCCESS':
            predict[i][0]=0
        else: predict[i][0]=1 
    category[:,0]=1+sa
    osCount={}
    for os in user['OPERATING_SYSTEM']:
        osCount[os]=osCount.get(os,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['OPERATING_SYSTEM'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][1]=osCount[user['OPERATING_SYSTEM'][user.index[i]]]/length
    oslen=len(osCount)
    ostmp=list(osCount)
    for i in range(0,length):
        for j in range(0,oslen):
            if user['OPERATING_SYSTEM'][user.index[i]]==ostmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][1]=j+1+sa
    deviceCount={}
    for dt in user['DEVICE_TYPE']:
        deviceCount[dt]=deviceCount.get(dt,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['DEVICE_TYPE'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][2]=deviceCount[user['DEVICE_TYPE'][user.index[i]]]/length
    dtlen=len(deviceCount)
    dttmp=list(deviceCount)
    for i in range(0,length):
        for j in range(0,dtlen):
            if user['DEVICE_TYPE'][user.index[i]]==dttmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][2]=j+1+sa
    browserCount={}
    for bc in user['BROWSER']:
        browserCount[bc]=browserCount.get(bc,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['BROWSER'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][3]=browserCount[user['BROWSER'][user.index[i]]]/length
    bclen=len(browserCount)
    bctmp=list(browserCount)
    for i in range(0,length):
        for j in range(0,bclen):
            if user['BROWSER'][user.index[i]]==bctmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][3]=j+1+sa
    connectionType={}
    for ct in user['client_connectionType']:
        connectionType[ct]=connectionType.get(ct,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['client_connectionType'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][4]=connectionType[user['client_connectionType'][user.index[i]]]/length
    ctlen=len(connectionType)
    cttmp=list(connectionType)
    for i in range(0,length):
        for j in range(0,ctlen):
            if user['client_connectionType'][user.index[i]]==cttmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][4]=j+1+sa
    appCount={}
    for an in user['APPLICATION_NAME']:
        appCount[an]=appCount.get(an,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['APPLICATION_NAME'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][5]=appCount[user['APPLICATION_NAME'][user.index[i]]]/length
    aclen=len(appCount)
    acnum=np.linspace(1,aclen,aclen)
    actmp=list(appCount)
    for i in range(0,length):
        for j in range(0,aclen):
            if user['APPLICATION_NAME'][user.index[i]]==actmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][5]=j+1+sa
    clientCountry={}
    for ccoun in user['client_country']:
        clientCountry[ccoun]=clientCountry.get(ccoun,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['client_country'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][6]=clientCountry[user['client_country'][user.index[i]]]/length
    ccounlen=len(clientCountry)
    ccountmp=list(clientCountry)
    for i in range(0,length):
        for j in range(0,ccounlen):
            if user['client_country'][user.index[i]]==ccountmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][6]=j+1+sa
    clientCity={}
    for ccity in user['client_city']:
        clientCity[ccity]=clientCity.get(ccity,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['client_city'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][7]=clientCity[user['client_city'][user.index[i]]]/length
    ccitylen=len(clientCity)
    ccitytmp=list(clientCity)
    for i in range(0,length):
        for j in range(0,ccitylen):
            if user['client_city'][user.index[i]]==ccitytmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][7]=j+1+sa
    clientISP={}
    for cisp in user['client_isp']:
        clientISP[cisp]=clientISP.get(cisp,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['client_isp'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][8]=clientISP[user['client_isp'][user.index[i]]]/length
    cisplen=len(clientISP)
    cisptmp=list(clientISP)
    for i in range(0,length):
        for j in range(0,cisplen):
            if user['client_isp'][user.index[i]]==cisptmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][8]=j+1+sa
    clientOrganization={}
    for co in user['client_organization']:
        clientOrganization[co]=clientOrganization.get(co,0.0)+1.0
    for i in range(0,length):
        if pd.isnull(user['client_organization'][user.index[i]]):
            user.fillna(method='bfill')
        predict[i][9]=clientOrganization[user['client_organization'][user.index[i]]]/length 
    colen=len(clientOrganization)
    cotmp=list(clientOrganization)
    for i in range(0,length):
        for j in range(0,colen):
            if user['client_organization'][user.index[i]]==cotmp[j]:
                sa=np.random.normal(mu,sigma)
                category[i][9]=j+1+sa

    
    feat_data = pd.DataFrame(predict, columns=columns_list)

    return category,user, feat_data

def add_flag(data_frame):
    os_list = np.array(data_frame['OPERATING_SYSTEM'].values)
    length = len(os_list)
    os_list = np.sort(os_list)
    os_threshold = os_list[length//2]
    flags = np.zeros((length))

    for i in range(length):
        if data_frame['OPERATING_SYSTEM'][data_frame.index[i]] >= os_threshold:
            flags[i] = 1
            
    data_frame = data_frame.assign(flag=flags)
    #data_frame.to_csv('sorted_sheet.csv', sep='\t')

    return shuffle(data_frame)