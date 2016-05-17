## 1. Introduction

### 1.1 Tensorflow

TensorFlow™ is an open source software library for numerical computation using data flow graphs. More information can be found here: <https://www.tensorflow.org/>

### 1.2 Goals of Creating This Repository

* Touch water of deep learning
* Get familiar with Tensorflow
* Review meachine learning algorithms
	* linear regression
	* k-mean clustering 
	* neural networks


## 2. Implementation

### 2.1 Linear Regression Using TensorFlow

Linear regression is one of simplest machine learning algorithms. The aim of it is to find a linear model that could best fit the given dataset.

#### 2.1.1 Create Data

```
# The number of points 
num_points = 1000
vectors_set = []
for i in xrange(num_points):
	# using numpy to create random points.
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])
```

![Fiture 1](https://github.com/AngeloK/tensorflow-demo/blob/master/static/random_points.png)

#### 2.1.2 Fit Data

```
w = tf.Variable(tf.constant(value=1.0))
b = tf.Variable(tf.constant(value=0.0))
```
These 2 lines create 2 parameters used for our linear model such that:

								y = Wx + b

Then we define loss function(aks cost function) as:

							h = mean(square(y - y_data))
							
By minimizing the cost function, we are able to find the optimal parameter w and b.

```
# Loss function
sse = tf.reduce_mean(tf.pow(tf.sub(y_, y_data), 2))

# We use gradient descent with learning rate equals to 0.5.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(sse)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    sess.run(train_step)
```
Pay attention here, in each tensorflow program, we have to create a session and run the data flow graph within the session. 

After running gradient descent 1000 times. we've got the optiaml w and b, which are [0.298826, 0.100289]

Then it't easy to plot a line on the top of the graph we created before. It is shown as follow:

![Fiture 2](https://github.com/AngeloK/tensorflow-demo/blob/master/static/linear_model.png)

By using tensorflow, linear regression could be easily implemented without writing too much code and importing scitific modules. And it's faster with high accuracy. 