# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:01:15 2018

@author: ljc
"""

import pandas as pd

data1 = pd.read_csv('break.csv',header=None)
data2 = pd.read_csv('drop.csv',header=None)
data3 = pd.read_csv('drop_har.csv',header=None)
data4 = pd.read_csv('flicker.csv',header=None)
data5 = pd.read_csv('harmonic.csv',header=None)
data6 = pd.read_csv('raise.csv',header=None)
data7 = pd.read_csv('raise_har.csv',header=None)
data8 = pd.read_csv('shock.csv',header=None)
data9 = pd.read_csv('sin.csv',header=None)



data = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8,data9],axis=1)
data = data.T
data.to_csv('concat.csv',header=None,index=False)