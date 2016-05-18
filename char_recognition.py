#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def read_data():
    data = pd.read_csv("train.csv")
    label = pd.read_csv("one_hot.csv")
    return data, label


def translate_to_one_hot(y):
    n = len(y)
    from collections import Counter
    columns = Counter(y).keys()
    one_hot = np.zeros((n, 62))
    for i in range(n):
        for col in columns:
            if y.ix[i] == col:
                one_hot[i, columns.index(col)] = 1
    df = pd.DataFrame(data=one_hot, columns=columns)
    return df


# Create placeholders for inputs and labels.
# None means it can be any value.
x = tf.placeholder(tf.float32, shape=[None, 400])

# There are 10 digits (0..9) and 52 chars (A..Z and a..z).
y_ = tf.placeholder(tf.float32, shape=[None, 62])

# Two helper functions to create parameters w and b.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")

# Bench size 5x5, the input is 1 channel and output is 32 channels.
W_cov1 = weight_variable([5, 5, 1, 64])
b_cov1 = bias_variable([64])

# The dimension of input of tf.nn.conv2d is
# [bench_size, image_width, image_height, output_channels]
# So we should reshape the image tensor.

x_reshaped = tf.reshape(x, [-1, 20, 20, 1])


h_cov1 = tf.nn.relu(conv2d(x_reshaped, W_cov1) + b_cov1)
h_pool1 = max_pool_2x2(h_cov1)

W_cov2 = weight_variable([5, 5, 64, 128])
b_cov2 = bias_variable([128])

h_cov2 = tf.nn.relu(conv2d(h_pool1, W_cov2) + b_cov2)
h_pool2 = max_pool_2x2(h_cov2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*128])

W_fc1 = weight_variable([5*5*128, 1024])
b_fc1 = bias_variable([1024])


h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# After craeting two convolutional layers, we use the neurons created before to
# construct a fully connected neural networks.
# Before it, we can use dropout to reduce the neurons with lower
# possibabilities.

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 62])
b_fc2 = bias_variable([62])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

portition_index = 0
bench = 50
predict = tf.argmax(y_conv, 1)

x_data, y_data = read_data()

# Create a new session
sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())
for i in range(1000):
    bench_start = portition_index * bench
    bench_end = bench_start + bench

    if y_data[bench_start: bench_start].shape[0] == 0:
        portition_index = 0
    if i % 100 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x: x_data.ix[bench_start: bench_end+1],
                         y_: y_data.ix[bench_start: bench_end+1],
                         keep_prob: 1.0
                         })
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train.run(feed_dict={x: x_data.ix[bench_start: bench_end+1],
                         y_: y_data.ix[bench_start: bench_end+1],
                         keep_prob: 0.5
                         })
    portition_index += 1

summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
test_x = x_data[500:520]

test_y = pd.read_csv("dataset/trainLabels.csv")["Class"][500:520]

print test_y

print "Prediction:"

columns = y_data.columns

prediction = predict.eval(feed_dict={
    x: test_x,
    keep_prob: 1.0
})

result = []

for i in prediction:
    result.append(columns[i])

print result
