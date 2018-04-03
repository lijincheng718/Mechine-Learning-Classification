# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:52:22 2018

@author: ljc
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'


#信噪比函数
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

plt.figure(figsize=(12,2))
#电压暂升

plt.subplot(121)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
plt.xticks([0,128,256,384,512,640],[0,0.04,0.08,0.12,0.16,0.2])
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Y,color='k')


#含谐波的电压暂升

plt.subplot(122)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Yhr,color='k')
plt.savefig('1.svg',bbox_inches='tight')


#电压暂降

plt.figure(figsize=(12,2))
plt.subplot(121)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Y,color='k')




#含谐波的电压暂降

plt.subplot(122)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Yhd,color='k')
plt.savefig('2.svg',bbox_inches='tight')


#电压中断
plt.figure(figsize=(12,2))
plt.subplot(121)
x = np.linspace(0,20*np.pi,640)
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
i = A*np.sin(x[int(64*t1):int(64*t2)])
yi = y[int(64*t1):int(64*t2)]-i
Y = np.row_stack((y[:int(64*t1)].reshape(-1,1),yi.reshape(-1,1)))
Y = np.row_stack((Y,y[int(64*t2):].reshape(-1,1)))
Y = Y+wgn(Y,30).reshape(-1,1)
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Y,color='k')



#电压闪变
plt.subplot(122)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,200)/1000
B = np.random.randint(5,20)                  
yf = (1+A*np.sin(B*x))*np.sin(x)
yf = yf+wgn(yf,30)
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(yf,color='k')
plt.savefig('3.svg',bbox_inches='tight')


#电压震荡

plt.figure(figsize=(12,2))
plt.subplot(121)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
Y = Y+wgn(Y,30).reshape(-1,1)
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Y,color='k')



#电压谐波

plt.subplot(122)
x = np.linspace(0,20*np.pi,640)
A1 = np.random.randint(20,200)/1000
A2 = np.random.randint(20,200)/1000
A3 = np.random.randint(20,200)/1000
h = np.sin(x)+A1*np.sin(3*x)+A2*np.sin(5*x)+A3*np.sin(7*x)
h = h+wgn(h,30)
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(x,h,color='k')
plt.savefig('4.svg',bbox_inches='tight')
plt.show()

















#正弦波
'''
plt.figure(figsize=(12,2))
x = np.linspace(0,20*np.pi,640)
y = np.sin(x)
y = y+wgn(y,30)
plt.plot(y)
plt.show()
'''

'''
plt.figure(figsize=(12,9))
#电压暂升

plt.subplot(421)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
plt.xticks([0,128,256,384,512,640],[0,0.04,0.08,0.12,0.16,0.2])
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Y,color='k')


#含谐波的电压暂升

plt.subplot(422)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
h = np.sin(x)+A1*np.sin(3*x)+A2*np.sin(5*x)+A3*np.sin(7*x)

Yhr = Y+h.reshape(-1,1)
Yhr = Yhr+wgn(Yhr,30).reshape(-1,1)
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Yhr,color='k')



#电压暂降

plt.subplot(423)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Y,color='k')




#含谐波的电压暂降

plt.subplot(424)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Yhd,color='k')



#电压中断

plt.subplot(425)
x = np.linspace(0,20*np.pi,640)
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
i = A*np.sin(x[int(64*t1):int(64*t2)])
yi = y[int(64*t1):int(64*t2)]-i
Y = np.row_stack((y[:int(64*t1)].reshape(-1,1),yi.reshape(-1,1)))
Y = np.row_stack((Y,y[int(64*t2):].reshape(-1,1)))
Y = Y+wgn(Y,30).reshape(-1,1)
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Y,color='k')



#电压闪变

plt.subplot(426)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,200)/1000
B = np.random.randint(5,20)                  
yf = (1+A*np.sin(B*x))*np.sin(x)
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(yf,color='k')



#电压震荡

plt.subplot(427)
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
Y = Y+wgn(Y,30).reshape(-1,1)
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(Y,color='k')



#电压谐波

plt.subplot(428)
x = np.linspace(0,20*np.pi,640)
A1 = np.random.randint(20,200)/1000
A2 = np.random.randint(20,200)/1000
A3 = np.random.randint(20,200)/1000
h = np.sin(x)+A1*np.sin(3*x)+A2*np.sin(5*x)+A3*np.sin(7*x)
h = h+wgn(h,30)
plt.xlabel('t/s')
plt.ylabel('amplitude')
plt.plot(x,h,color='k')
plt.savefig('PQD仿真.png',bbox_inches='tight')
plt.show()
'''







'''
#信噪比函数
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

#正弦波
plt.figure(figsize=(12,2))
x = np.linspace(0,20*np.pi,640)
y = np.sin(x)
y = y+wgn(y,30)
plt.plot(y)
plt.show()

#电压暂升

plt.figure(figsize=(12,2))
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
plt.plot(Y)
plt.show()


#含谐波的电压暂升

plt.figure(figsize=(12,4))
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
h = np.sin(x)+A1*np.sin(3*x)+A2*np.sin(5*x)+A3*np.sin(7*x)

Yhr = Y+h.reshape(-1,1)
Yhr = Yhr+wgn(Yhr,30).reshape(-1,1)
plt.plot(Yhr)
plt.show()


#电压暂降

plt.figure(figsize=(12,2))
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
plt.plot(Y)
plt.show()


#含谐波的电压暂升

plt.figure(figsize=(12,2))
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
h = np.sin(x)+A1*np.sin(3*x)+A2*np.sin(5*x)+A3*np.sin(7*x)
Yhd = Y+h.reshape(-1,1)
Yhd = Yhd+wgn(Yhd,30).reshape(-1,1)
plt.plot(Yhd)
plt.show()


#电压中断

plt.figure(figsize=(12,2))
x = np.linspace(0,20*np.pi,640)
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
i = A*np.sin(x[int(64*t1):int(64*t2)])
yi = y[int(64*t1):int(64*t2)]-i
Y = np.row_stack((y[:int(64*t1)].reshape(-1,1),yi.reshape(-1,1)))
Y = np.row_stack((Y,y[int(64*t2):].reshape(-1,1)))
Y = Y+wgn(Y,30).reshape(-1,1)
plt.plot(Y)
plt.show()


#电压闪变

plt.figure(figsize=(12,2))
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,200)/1000
B = np.random.randint(5,20)                  
yf = (1+A*np.sin(B*x))*np.sin(x)
plt.plot(yf)
plt.show()


#电压震荡

plt.figure(figsize=(12,2))
x = np.linspace(0,20*np.pi,640)
A = np.random.randint(100,900)/1000
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
Y = Y+wgn(Y,30).reshape(-1,1)
plt.plot(Y)
plt.show()


#电压谐波

plt.figure(figsize=(12,2))
x = np.linspace(0,20*np.pi,640)
A1 = np.random.randint(20,200)/1000
A2 = np.random.randint(20,200)/1000
A3 = np.random.randint(20,200)/1000
h = np.sin(x)+A1*np.sin(3*x)+A2*np.sin(5*x)+A3*np.sin(7*x)
h = h+wgn(h,30)
plt.plot(x,h)
plt.show()
'''