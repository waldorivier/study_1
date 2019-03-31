import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures

#--------------------------------------------------------------------------
a = b = c = 1
def f_(a, b, c, x1, x2):
    return (a * x1 +  b * x2 + c)
    
# prepare train data 

X = []
y = []

for i in np.arange(0, 10):
    for j in np.arange(0, 10):
        yy = f_(a, b, c, i, j)
        X.append(xx)
        y.append(yy)

X = np.array(X)
y = np.c_[y]
    
#--------------------------------------------------------------------------

a_ = tf.Variable(initial_value=0.0, dtype=tf.float32)
b_ = tf.Variable(initial_value=0.0, dtype=tf.float32)
c_ = tf.Variable(initial_value=0.0, dtype=tf.float32)

X_ = tf.placeholder(dtype=tf.float32, shape=[X.shape[0], 2])
W_ = tf.Variable(initial_value=tf.zeros(shape=[2, 1]))

y_ = tf.matmul(X_, W_) + b_

loss = tf.reduce_mean( # Equivalent to np.mean()
    tf.square( # Equivalent to np.square()
        y_ - y
    )
)

lr_ = tf.placeholder(dtype=tf.float32)
gd = tf.train.GradientDescentOptimizer(
    learning_rate=lr_)

train_op = gd.minimize(loss)

initialization_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # Initialize the graph
    sess.run(initialization_op)

    # Compute predictions
    result = sess.run([train_op, loss], feed_dict={
        X_  : X, 
        y_  : y, 
        lr_ : 0.1
    })

    W_fitt = W_.eval()
    print(result)



#------------------------------------------------------------------------------

