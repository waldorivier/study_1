import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures

#--------------------------------------------------------------------------
def get_batches(X, y, bulk):
    for i in range(0, len(y), bulk):
        yield X[i:i + bulk], y[i:i + bulk]

#--------------------------------------------------------------------------

w1 = w2 = b = 2
def f_(x1, x2, w1, w2, b):
    return (w1 * x1 ** 2 +  w2 * x2 + b)
    
# prepare train data 

X = []
y = []

for i in np.arange(0, 10):
    for j in np.arange(0, 10):
        yy = f_(i, j, w1, w2, b)
        xx = [i,j]
        X.append(xx)
        y.append(yy)

X = np.array(X)
y = np.c_[y]
    
#--------------------------------------------------------------------------

X_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1])
b_ = tf.Variable(initial_value=0.0, dtype=tf.float32)

W_ = tf.Variable(initial_value=tf.zeros(shape=[2, 1]))
y_est = tf.matmul(X_, W_) + b_

loss = tf.reduce_mean( # Equivalent to np.mean()
    tf.square( # Equivalent to np.square()
        y_ - y_est
    )
)

lr_ = tf.placeholder(dtype=tf.float32)
gd = tf.train.GradientDescentOptimizer(
    learning_rate=lr_)

train_op = gd.minimize(loss)

l_loss = []
initialization_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # Initialize the graph
    sess.run(initialization_op)

    for X_b, y_b  in get_batches(X, y, 1):
        _, l = sess.run([train_op, loss], feed_dict={
                        X_  : X_b, 
                        y_  : y_b, 
                        lr_ : 0.001
        })

        l_loss.append(l)
        W_fitt = W_.eval()
        b_fitt = b_.eval()
       
    
plt.plot(l_loss)
plt.show()

#------------------------------------------------------------------------------

