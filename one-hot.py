# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:12:37 2018

@author: ljc
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

# 整数类别
l = [i for i in range(1,10)]
df = pd.DataFrame(np.random.randn(1000,9))
for i in range(9):
    df[i] = l[i]
df.to_csv('target.csv',header=['break','drop','drop_har','flicker','harmonic','raise','raise_har','shock','sin'],index=False)

# one-hot
enc = preprocessing.OneHotEncoder()
enc.fit([[1], [2], [3], [4],[5],[6],[7],[8],[9]])  
oh = enc.transform(df.values.T.reshape(-1,1)).toarray()
df1 = pd.DataFrame(oh)
df1.to_csv('target_oh.csv',header=None,index=False)

 
