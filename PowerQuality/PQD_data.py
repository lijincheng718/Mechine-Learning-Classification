# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:27:02 2018

@author: ljc
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd


n = 1000    #样本数
m = 400     #频率
x = np.linspace(0,20*np.pi,m)
df = pd.DataFrame(np.random.randn(m,n))

#信噪比函数
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

#正弦波
for i in range(n):
    y = np.sin(x)
    y = y+wgn(y,30)
    df[i] = y.reshape(-1,1)
df.to_csv('sin.csv',header=None,index=False)


#电压暂升
for i in range(n):
    A = np.random.randint(200,900)/1000
    t1 = np.random.randint(0,900)/100
    t2 = np.random.randint(100,1000)/100
                          
    if t1>t2:
        temp=t2
        t2=t1
        t1=temp
    if (t2-t1)<1:
        t1=np.random.randint(0,400)/100
        t2=np.random.randint(700,900)/100                    
                          
    y = np.sin(x)
    r = A*np.sin(x[int(64*t1):int(64*t2)])
    yr = r+y[int(64*t1):int(64*t2)]
    Y = np.row_stack((y[:int(64*t1)].reshape(-1,1),yr.reshape(-1,1)))
    Y = np.row_stack((Y,y[int(64*t2):].reshape(-1,1)))
    Y = Y+wgn(Y,30).reshape(-1,1)
    df[i] = Y
df.to_csv('raise.csv',header=None,index=False)
    

#含谐波的电压暂升
for i in range(n):
    A = np.random.randint(200,900)/1000
    t1 = np.random.randint(0,900)/100
    t2 = np.random.randint(100,1000)/100
                          
    if t1>t2:
        temp=t2
        t2=t1
        t1=temp
    if (t2-t1)<1:
        t1=np.random.randint(0,400)/100
        t2=np.random.randint(700,900)/100                    
                          
    y = np.sin(x)
    r = A*np.sin(x[int(64*t1):int(64*t2)])
    yr = r+y[int(64*t1):int(64*t2)]
    Y = np.row_stack((y[:int(64*t1)].reshape(-1,1),yr.reshape(-1,1)))
    Y = np.row_stack((Y,y[int(64*t2):].reshape(-1,1)))
    
    A1 = np.random.randint(20,200)/1000
    A2 = np.random.randint(20,200)/1000
    A3 = np.random.randint(20,200)/1000
    h = A1*np.sin(3*x)+A2*np.sin(5*x)+A3*np.sin(7*x)
    
    Yhr = Y+h.reshape(-1,1)
    Yhr = Yhr+wgn(Yhr,30).reshape(-1,1)
    df[i] = Yhr
df.to_csv('raise_har.csv',header=None,index=False)


#电压暂降
for i in range(n):
    A = np.random.randint(200,800)/1000
    t1 = np.random.randint(0,900)/100
    t2 = np.random.randint(100,1000)/100
                          
    if t1>t2:
        temp=t2
        t2=t1
        t1=temp
    if (t2-t1)<1:
        t1=np.random.randint(0,400)/100
        t2=np.random.randint(700,900)/100                    
                             
    y = np.sin(x)
    d = A*np.sin(x[int(64*t1):int(64*t2)])
    yd = y[int(64*t1):int(64*t2)]-d
    Y = np.row_stack((y[:int(64*t1)].reshape(-1,1),yd.reshape(-1,1)))
    Y = np.row_stack((Y,y[int(64*t2):].reshape(-1,1)))
    Y = Y+wgn(Y,30).reshape(-1,1)
    df[i] = Y
df.to_csv('drop.csv',header=None,index=False)


#含谐波的电压暂降
for i in range(n):
    A = np.random.randint(200,800)/1000
    t1 = np.random.randint(0,900)/100
    t2 = np.random.randint(100,1000)/100
                          
    if t1>t2:
        temp=t2
        t2=t1
        t1=temp
    if (t2-t1)<1:
        t1=np.random.randint(0,400)/100
        t2=np.random.randint(700,900)/100                    
                          
    y = np.sin(x)
    d = A*np.sin(x[int(64*t1):int(64*t2)])
    yd = y[int(64*t1):int(64*t2)]-d
    Y = np.row_stack((y[:int(64*t1)].reshape(-1,1),yd.reshape(-1,1)))
    Y = np.row_stack((Y,y[int(64*t2):].reshape(-1,1)))
    
    A1 = np.random.randint(20,200)/1000
    A2 = np.random.randint(20,200)/1000
    A3 = np.random.randint(20,200)/1000
    h = A1*np.sin(3*x)+A2*np.sin(5*x)+A3*np.sin(7*x)
    Yhd = Y+h.reshape(-1,1)
    Yhd = Yhd+wgn(Yhd,30).reshape(-1,1)
    df[i] = Yhd
df.to_csv('drop_har.csv',header=None,index=False)


#电压中断
for i in range(n):
    A = np.random.randint(900,1000)/1000
    t1 = np.random.randint(0,900)/100
    t2 = np.random.randint(100,1000)/100
                          
    if t1>t2:
        temp=t2
        t2=t1
        t1=temp
    if (t2-t1)<1:
        t1=np.random.randint(0,400)/100
        t2=np.random.randint(700,900)/100                    
                          
    y = np.sin(x)
    b = A*np.sin(x[int(64*t1):int(64*t2)])
    yb = y[int(64*t1):int(64*t2)]-b
    Y = np.row_stack((y[:int(64*t1)].reshape(-1,1),yb.reshape(-1,1)))
    Y = np.row_stack((Y,y[int(64*t2):].reshape(-1,1)))
    Y = Y+wgn(Y,30).reshape(-1,1)
    df[i] = Y
df.to_csv('break.csv',header=None,index=False)


#电压闪变
for i in range(1000):
    A = np.random.randint(100,200)/1000
    B = np.random.randint(5,20)                  
    yf = (1+A*np.sin(B*x))*np.sin(x)
    yf = yf+wgn(yf,30)
    df[i] = yf.reshape(-1,1)
df.to_csv('flicker.csv',header=None,index=False)


#电压震荡
for i in range(n):
    A = np.random.randint(200,900)/1000
    B = np.random.randint(500,800)/100
    t1 = np.random.randint(0,950)/100
    t2 = np.random.randint(50,1000)/100
    tao = np.random.randint(250,1300)/1000                    
    if t1>t2:
        temp=t2
        t2=t1
        t1=temp
    if (t2-t1)<0.5:
        t1=np.random.randint(0,100)/100
        t2=np.random.randint(200,300)/100                    
    if (t2-t1)>3:
        t1=np.random.randint(700,800)/100
        t2=np.random.randint(900,1000)/100                       
    
    y = np.sin(x)
    s = A*np.sin(B*x[int(64*t1):int(64*t2)])*np.exp((x[int(64*t1):int(64*t2)])*tao*-1/640)
    ys = s+y[int(64*t1):int(64*t2)]
    Y = np.row_stack((y[:int(64*t1)].reshape(-1,1),ys.reshape(-1,1)))
    Y = np.row_stack((Y,y[int(64*t2):].reshape(-1,1)))
    Y = Y+wgn(Y,30)
    df[i] = Y
df.to_csv('shock.csv',header=None,index=False)


#电压谐波
for i in range(n):
    A1 = np.random.randint(20,200)/1000
    A2 = np.random.randint(20,200)/1000
    A3 = np.random.randint(20,200)/1000
    h = np.sin(x)+A1*np.sin(3*x)+A2*np.sin(5*x)+A3*np.sin(7*x)
    h = h+wgn(h,30)
    df[i] = h.reshape(-1,1)
df.to_csv('harmonic.csv',header=None,index=False)
