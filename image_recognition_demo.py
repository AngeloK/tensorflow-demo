#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Source:
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners


# Fetch data from MNIST, 1-d vector with a particular position set to 1 and other
# places are set to 0 is used if one_hot is set to True.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Create a new session
sess = tf.InteractiveSession()

# Placeholders for variables x and y
# 'None' means it can be any value
# In MNIST image dataset, images are stored as 1-d vector of size 1 x 784.
# Since there are 10 digits (0..9), the size of output vectors are 1 x 10.
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # x is a 4d tensor input, W is filter.
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


# The first two parameters are the size of batch, in this example 5 x 5
# '1' is the input channel, '32' is the output channel.
# These 2 statments created 32 feature mappings from the original image.
W_cov1 = weight_variable([5, 5, 1, 32])
b_cov1 = bias_variable([32])


# Since the tf.nn.conv2d function needs the first paraemter to be a tensor which
# defined the size of data with 28 x 28, so we reshape the original image to
# 28 x 28 with 1 as the output channel.
x_reshaped = tf.reshape(x, [-1, 28, 28, 1])


# Compute the first convulution layer
h_cov1 = tf.nn.relu(conv2d(x_reshaped, W_cov1) + b_cov1)

# subsampling, here we use max of 2x2 batch.
h_pool1 = max_pool_2x2(h_cov1)

# Create filters and biases for the second convolution layer.
W_cov2 = weight_variable([5, 5, 32, 64])
b_cov2 = bias_variable([64])

# Create the second convolution layer.
h_cov2 = tf.nn.relu(conv2d(h_pool1, W_cov2) + b_cov2)

# subsampling again.
h_pool2 = max_pool_2x2(h_cov2)


# Craete a fully connected neural networks using the ouput of the 2 convolution
# layers
W_fc1 = weight_variable([7 * 7 * 64, 1024])

# define 1024 as the number of neurons.
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# TODO this is the part I don't understand.
# There is a possibability for each neuron, such that if the possibability of
# a neuron is less than 0.5, the neuron and all connections related to the
# neuron is deleted.
# This could reduce the computation and speed up the program.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Basicaly, these 2 statements above would simplify the dataset and speed up the
# whole program.

# Noraml neural networks computataion.
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Compute the results by using softmax.
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define the loss function(or cost function), here we use y_hat * log(y).
cross_entropy = tf.reduce_mean(-
                               tf.reduce_sum(y_ *
                                             tf.log(y_conv), reduction_indices=[1]))

# Using Adamoptimizer to compute the parameters.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Test prediction
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run program.
sess.run(tf.initialize_all_variables())
for i in range(2000):
    batch = mnist.train.next_batch(50)
    # if i % 100 == 0:
        # train_accuracy = accuracy.eval(feed_dict={
            # x: batch[0], y_: batch[1], keep_prob: 1.0})
        # print("step %d, training accuracy %g" % (i, train_accuracy))
    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# print("test accuracy %g" % accuracy.eval(feed_dict={
    # x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
