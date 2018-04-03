# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:34:48 2018

@author: ljc
"""

import time
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_csv('concat.csv',header=None)
data = data.values
target = pd.read_csv('target_oh.csv',header=None)
target = target.values
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=0)
print(X_train.shape[0])
train_size = X_train.shape[0]

batch_size = 64
out1 = 32
out2 = 64
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1],Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    #最大池化函数
    #return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')   #平均池化函数(效果不如max函数)


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 400])   # 20x20
ys = tf.placeholder(tf.float32, [None, 9])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 20, 20, 1])
# print(x_image.shape)  # [n_samples, 20,20,1]

## conv1 layer 
W_conv1 = weight_variable([5,5, 1,out1])    # 卷积核patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([out1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 20x20x32
h_pool1 = max_pool_2x2(h_conv1)                          # output size 10x10x32

## conv2 layer 
W_conv2 = weight_variable([5,5, out1, out2])    # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([out2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                          # output size 5x5x64

## fc1 layer ##
W_fc1 = weight_variable([5*5*out2, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 5, 5, 64] ->> [n_samples, 5*5*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*out2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 9])
b_fc2 = bias_variable([9])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

t1 = time.time()
# important step
init = tf.global_variables_initializer()
sess.run(init)
for j in range(20):
    for i in range(int(train_size/batch_size)):
        batch_xs, batch_ys = X_train[batch_size*i:batch_size*(i+1)],y_train[batch_size*i:batch_size*(i+1)]
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 40 == 0:
            print(compute_accuracy(
                X_test[:], y_test[:]))

t2 = time.time()
print(t2-t1)
