# encoding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

print(FLAGS.data_dir)

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

with tf.name_scope('layer1'):
    W_conv1 = weight_variable([3, 3, 1, 32])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)

    h_pool1 = max_pool_2x2(h_conv1)


# 第二层

with tf.name_scope('layer2'):
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)


with tf.name_scope("layer3"):
    W_conv3=weight_variable([3,3,64,64])
    b_conv3=bias_variable([64])
    h_conv3=tf.nn.relu(conv2d(h_conv2,W_conv3)+b_conv3)

    #mean,variance=tf.nn.moments(h_pool2,[0])

 #batch_norm_2=tf.nn.batch_normalization(h_pool2,mean,variance,0,1,0.1)
with tf.name_scope("layer4"):
    W_conv4=weight_variable([3,3,64,64])
    b_conv4=bias_variable([64])
    h_conv4=tf.nn.relu(conv2d(h_conv3,W_conv4)+b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)
"""
with tf.name_scope("layer5"):
    W_conv5=weight_variable([3,3,64,128])
    b_conv5=bias_variable([128])
    h_conv5=tf.nn.relu(conv2d(h_conv4,W_conv5)+b_conv5)
with tf.name_scope("layer6"):
    W_conv6=weight_variable([3,3,128,128])
    b_conv6=bias_variable([128])
    h_conv6=tf.nn.relu(conv2d(h_conv5,W_conv6)+b_conv6)
with tf.name_scope("layer7"):
    W_conv7=weight_variable([3,3,128,256])
    b_conv7=bias_variable([256])
    h_conv7=tf.nn.relu(conv2d(h_conv6,W_conv7)+b_conv7)
    h_pool7=max_pool_2x2(h_conv7)
"""
with tf.name_scope('dropout'):
    W_fc1 = weight_variable([7 * 7* 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool4, [-1, 7* 7* 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32) #
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#softmax
with tf.name_scope('softmax'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y_ = tf.placeholder(tf.float32, [None, 10])
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # 损失函数，交叉熵
    tf.summary.scalar('cross_entropy',cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

with tf.name_scope('evaluate'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # 计算准确度
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

sess.run(tf.global_variables_initializer())



merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter('/tmp/mnist_wy',sess.graph)


for i in range(5000):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:


        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})


        print("step %d, training accuracy %g"%(i, train_accuracy))
    run_option=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    summary,_=sess.run([merged,train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5},options=run_option)
    train_writer.add_summary(summary,i)

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
train_writer.close()