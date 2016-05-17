#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

num_points = 1000
vectors_set = []
for i in xrange(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = np.array([v[0] for v in vectors_set])
y_data = np.array([v[1] for v in vectors_set])

df = pd.DataFrame({"x": x_data, "y": y_data})


sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.legend()
# plt.show()


w = tf.Variable(tf.constant(value=1.0))
b = tf.Variable(tf.constant(value=0.0))

y_ = tf.add(tf.mul(w, x_data), b)
# y_ = w * x_data + b


test = tf.sub(y_, y_data)

# sse = tf.reduce_mean(tf.square(y_ - y_data))
sse = tf.reduce_mean(tf.pow(tf.sub(y_, y_data), 2))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(sse)

sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())

for i in range(1000):
    sess.run(train_step)

plt.plot(x_data, sess.run(w) * x_data + sess.run(b))
plt.show()
