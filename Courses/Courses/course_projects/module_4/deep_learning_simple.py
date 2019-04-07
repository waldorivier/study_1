import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


#------------------------------------------------------------------------------
# plot 3D surface
#------------------------------------------------------------------------------
def plot_surface(f, x1, x2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # calculate data
    x1_, x2_ = np.meshgrid(x1, x2)
    z = f(x1_, x2_)

    surf = ax.plot_surface(x1, x2, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(0, z.max())
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()

#--------------------------------------------------------------------------
def get_batches(X, y, bulk):
    for i in range(0, len(y), bulk):
        yield X[i:i + bulk], y[i:i + bulk]

#--------------------------------------------------------------------------

w1 = 1
w2 = 1
b = 1

def f(w1, w2, b):
    w1_ = w1
    w2_ = w2
    b_ = b

    def f_(x1, x2):
        return (w1_ * x1 +  w2_ * x2 + b_)
        # return (w1_ * x1 ** 2 +  w2_ * x2 ** 2 + b_)
    
    return f_

#--------------------------------------------------------------------------
# generates pseudo data
# prepare train data 

f_pseudo = f(w1, w2, b)

X = []
y = []

for i in np.arange(0, 10, 0.1):
    for j in np.arange(0, 10, 0.1):
        yy = f_pseudo(i, j)
        xx = [i,j]
        X.append(xx)
        y.append(yy)

X = np.array(X)
y = np.c_[y]
    
#--------------------------------------------------------------------------
# defines tensor graph 
#--------------------------------------------------------------------------
scaler = StandardScaler()

if 0:
    X_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    b_ = tf.Variable(initial_value=0.0, dtype=tf.float32)
    W_ = tf.Variable(initial_value=tf.zeros(shape=[2, 1]))

    X_2_ = tf.pow(X_, 1)
    y_est_ = tf.matmul(X_2_, W_) + b_

    loss = tf.reduce_mean( # Equivalent to np.mean()
        tf.square( # Equivalent to np.square()
            y_ - y_est_
        )
    )

    lr_ = tf.placeholder(dtype=tf.float32)
    gd = tf.train.GradientDescentOptimizer(
        learning_rate=lr_)

    train_op = gd.minimize(loss)

    # train nn model

    scaler.fit(X)
    X_s = scaler.transform(X)

    l_loss = []
    initialization_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Initialize the graph
        sess.run(initialization_op)
      
        for X_b, y_b in get_batches(X_s, y, 50):
            _, l = sess.run([train_op, loss], feed_dict={
                            X_  : X_b, 
                            y_  : y_b, 
                            lr_ : 0.01
            })

            l_loss.append(l)
            w_fitt = W_.eval()
            b_fitt = b_.eval()
    
        # do predictions 
        y_est = sess.run(y_est_, feed_dict={
            X_ : X # Feed body weights
        })
     
    f_estimate = f(w_fitt[0], w_fitt[1], b_fitt)
    plot_surface(f_estimate, np.arange (0, 10, 0.1), np.arange (0, 10, 0.1))
    plot_surface(f_pseudo, np.arange (0, 10, 0.1), np.arange (0, 10, 0.1))
    plt.show()

#------------------------------------------------------------------------------
# try adding one hidden layer
#------------------------------------------------------------------------------






