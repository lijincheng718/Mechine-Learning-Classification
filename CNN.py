# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:19:43 2018

@author: ljc
"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


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
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
init = tf.global_variables_initializer()
sess.run(init)

t1 = time.time()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(128)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))

t2 = time.time()
print(t2-t1)



'''


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())
#一次性初始化所有变量

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


#CNN

#我们会建立大量权重和偏置项，为了方便，定义初始函数


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) #tf.truncated_normal初始函数将根据所得到的均值和标准差，生成一个随机分布
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
"""
由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，
以避免神经元节点输出恒为0的问题（dead neurons）
"""

#卷积池化操作  
"""
我们的卷积使用1步长（stride size），0边距（padding size）的模板
保证输出和输入是同一个大小。我们的池化用简单传统的2x2大小的模板做max pooling
"""

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
"""
1. x是输入的样本，在这里就是图像。x的shape=[batch, height, width, channels]。 
- batch是输入样本的数量 
- height, width是每张图像的高和宽 
- channels是输入的通道，比如初始输入的图像是灰度图，那么channels=1，如果是rgb，那么channels=3。对于第二层卷积层，channels=32。 
2. W表示卷积核的参数，shape的含义是[height,width,in_channels,out_channels]。 
3. strides参数表示的是卷积核在输入x的各个维度下移动的步长。了解CNN的都知道，在宽和高方向stride的大小决定了卷积后图像的size。这里为什么有4个维度呢？因为strides对应的是输入x的维度，所以strides第一个参数表示在batch方向移动的步长，第四个参数表示在channels上移动的步长，这两个参数都设置为1就好。重点就是第二个，第三个参数的意义，也就是在height于width方向上的步长，这里也都设置为1。 
4. padding参数用来控制图片的边距，’SAME’表示卷积后的图片与原图片大小相同，’VALID’的话卷积以后图像的高为Heightout=Height原图−Height卷积核+1StrideHeight， 宽也同理。
"""

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
"""                       
这里用2∗2的max_pool。参数ksize定义pool窗口的大小，每个维度的意义与之前的strides相同
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
这个函数的功能是将整个图片分割成2x2的块，
对每个块提取出最大值输出。可以理解为对整个图片做宽度减小一半，高度减小一半的降采样
"""

#卷积一
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
"""
它由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征（理解为做了32次卷积，每次卷积中不同的神经元享有同样参数；但是不同次卷积所用的参数是不同的）。
卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，
最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量
"""


x_image = tf.reshape(x, [-1,28,28,1])
"""
为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，
最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
第一维-1代表将x沿着最后一维进行变形
"""

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#卷积二
"""
为了构建一个更深的网络，我们会把几个类似的层堆叠起来。
第二层中，每个5x5的patch会得到64个特征。
"""
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#密集连接层
"""
现在，图片尺寸减小到7x7（pool两次，相当于降采样两次），我们加入一个有1024个神经元的全连接层，
用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，
然后对其使用ReLU
"""
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#droput
"""
为了减少过拟合，我们在输出层之前加入dropout。
我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 
TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，
还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale

Dropout是指在模型训练时随机让网络某些隐含层节点的权重不工作，
不工作的那些节点可以暂时认为不是网络结构的一部分，
但是它的权重得保留下来（只是暂时不更新而已），
因为下次样本输入时它可能又得工作了
"""
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(500):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print ("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
'''