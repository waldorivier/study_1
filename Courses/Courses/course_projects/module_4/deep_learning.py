import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
import  PIL
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#------------------------------------------------------------------------------
pd.set_option('display.max_columns', 90)
sns.set_palette(sns.color_palette("hls", 20))

working_dir = os.getcwd()
working_dir = os.path.join(working_dir,'course_projects', 'module_4')

def format_number_image(min, max):
    # original image

    file_name = 'number_waldo.jpg'
    data_file = os.path.join(working_dir, file_name)

    img = Image.open(data_file)

    # manually select portion 
    # height = width = 500
    # area = (left, top, left + width, top + height)

    area = (1500, 750, 2000, 1250)
    reduced_img = img.crop(area)
    reduced_img = reduced_img.resize((28, 28))

    # rotation (anti-clockwise)
    reduced_img = reduced_img.rotate(-90)

    # convert to one-dimension 
    reduced_img = reduced_img.convert('L')

    a_reduced_img = np.array(reduced_img)
    l_filt_img = [ i + 50 if(i >= min and i < max) else 0 for i in a_reduced_img.flatten()]
    reduced_img = np.reshape(l_filt_img, (28,28))

    # save it 
    file_name = 'number_waldo.png'
    data_file = os.path.join(working_dir, file_name)
    np.save(data_file, reduced_img)
    
    return  np.array(reduced_img)

#------------------------------------------------------------------------------
file_name = 'brain-body-weights.csv'
data_file = os.path.join(working_dir, file_name)

data_df = pd.read_csv(data_file)

a = tf.Variable(initial_value=0, dtype=tf.float32)
b = tf.Variable(initial_value=0, dtype=tf.float32)

x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

# estimator
y_hat = a*x + b

# erreur == différnce entre estimateur et valeur attendure
loss = tf.reduce_mean( # Equivalent to np.mean()
    tf.square( # Equivalent to np.square()
        y - y_hat # Implements broadcasting like Numpy
    )
)

loss = tf.losses.huber_loss(y, y_hat, delta=1.0)

lr = tf.placeholder(dtype=tf.float32)
gd = tf.train.GradientDescentOptimizer(
    learning_rate=lr)

train_op = gd.minimize(loss)

initialization_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # Initialize the graph
    sess.run(initialization_op)

    # Compute predictions
    result = sess.run([train_op, loss], feed_dict={
        x: data_df.body, # Body weights
        y: data_df.brain, # Brain weights
        lr : 0.1
    })
    print(result)

#------------------------------------------------------------------------------

# resolution = 3072
resolution = (784)

# file_name = 'mnist-6k.npz'
file_name = 'cifar10-6k.npz'
data_file = os.path.join(working_dir, file_name)

with np.load(data_file, allow_pickle=False) as npz_file:
    data = npz_file['data']
    labels = npz_file['labels']
    
X_train, X_test, y_train, y_test = train_test_split(
    # Convert uint8 pixel values to float
    data.astype(np.float32),
    labels,
    test_size=1000, random_state=0
)

scaler = StandardScaler()
scaler.fit(X_train)

# Split again into validation/test sets
X_valid, X_test, y_valid, y_test = train_test_split(
    X_test, y_test,
    test_size=500, random_state=0
)

X = tf.placeholder(dtype=tf.float32, shape=[None, resolution])
y = tf.placeholder(dtype=tf.int32, shape=[None])

W = tf.Variable(initial_value=tf.zeros(shape=[resolution, 10]))
b = tf.Variable(initial_value=tf.zeros(shape=[10]))

#------------------------------------------------------------------------------
# objective  function
#------------------------------------------------------------------------------
logits = tf.matmul(X, W) + b

#------------------------------------------------------------------------------
# loss function
#------------------------------------------------------------------------------
y_one_hot = tf.one_hot(indices=y, depth=10)
ce = tf.nn.softmax_cross_entropy_with_logits(
    labels=y_one_hot, # Requires one-hot encoded labels
    logits=logits
)

#------------------------------------------------------------------------------
# Optimizer
#------------------------------------------------------------------------------
mean_ce = tf.reduce_mean(ce)
lr = tf.placeholder(dtype=tf.float32, shape=[])
gd = tf.train.GradientDescentOptimizer(
    learning_rate=lr)

# Minimize cross-entropy
train_op = gd.minimize(mean_ce)

#------------------------------------------------------------------------------
# Predictor
#------------------------------------------------------------------------------
predictions = tf.argmax(
    logits, # shape: (n, 10)
    axis=1, # class with max logit
    output_type=tf.int32 # Same type as labels
)

#------------------------------------------------------------------------------
# Evaluator
#------------------------------------------------------------------------------
is_correct = tf.equal(y, predictions)

#------------------------------------------------------------------------------
# Accuracy
#------------------------------------------------------------------------------
accuracy = tf.reduce_mean(
    # Convert booleans (false/true) to 0/1 float numbers
    tf.cast(is_correct, dtype=tf.float32)
)

#------------------------------------------------------------------------------
# Batcher
#------------------------------------------------------------------------------
def get_batches(X, y, batch_size):
    # Enumerate indexes by steps of batch_size
    # i: 0, b, 2b, 3b, 4b, .. where b is the batch size
    for i in range(0, len(y), batch_size):
        # "yield" data between index i and i+b (not included)
        yield X[i:i+batch_size], y[i:i+batch_size]

#------------------------------------------------------------------------------
# Train
#------------------------------------------------------------------------------
acc_values = []

# Initialization operation
initialization_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initialize the graph
    sess.run(initialization_op)

    X_train_s = scaler.transform(X_train)
    # X_train_s = X_train

    # Get batches of data
    for X_batch, y_batch in get_batches(X_train_s, y_train, 64):
        # Run training and evaluate accuracy
        _, batch_acc = sess.run([train_op, accuracy], feed_dict={
            X: X_batch,
            y: y_batch,
            lr: 0.01 # learning rate
        })
        acc_values.append(batch_acc)

    # Get weight matrix and biases
    W_fitted = W.eval()
    b_fitted = b.eval()
    # .. which is equivalent to
    W_fitted, b_fitted = sess.run([W, b])

plt.plot(acc_values)
plt.title('Train accuracy (last 20 batches): {:.3f}'.format(
    # Average accuracy value
    np.mean(acc_values[-20:])
))
plt.xlabel('batch')
plt.ylabel('accuracy')
plt.show()

#------------------------------------------------------------------------------
# try to predict my number..
#------------------------------------------------------------------------------
a_img = format_number_image(125,140)
plt.imshow(a_img)
plt.show()

x_values = a_img.reshape(1,-1)

with tf.Session() as sess:
    # Initialize the graph
    sess.run(initialization_op)

    y_values = sess.run([logits, predictions], feed_dict={
            X: x_values, # Sample body weights
            W: W_fitted, 
            b: b_fitted 
        })

#------------------------------------------------------------------------------
# cifar animals recognition / classification
#------------------------------------------------------------------------------
W_fitted_rescaled = MinMaxScaler().fit_transform(W_fitted)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
flat_axes = axes.flatten() # from shape (2,5) to (10,)

# Plot the weights for each class in a subplot
for i, axis in enumerate(flat_axes):

        # Get weights of the i-th class
        weights = W_fitted_rescaled[:, i]

        # Reshape weight values into a 32x32 array
        template = weights.reshape(32, 32, 3)

        # Plot array
        axis.imshow(template)
        axis.get_xaxis().set_visible(False) # disable x-axis
        axis.get_yaxis().set_visible(False) # disable y-axis

plt.show()

#------------------------------------------------------------------------------
# MULTI LAYER NET-NEURONES
#------------------------------------------------------------------------------

# Create a new graph
graph = tf.Graph()

with graph.as_default():
    # Create placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.int32, shape=[None])

    # Hidden layer with 16 units
    W1 = tf.Variable(initial_value=tf.truncated_normal(
        shape=[784, 16], # Shape
        stddev=(2/784)**0.5, # Calibrating variance
        seed=0
    ))
    b1 = tf.Variable(initial_value=tf.zeros(shape=[16]))

    # Output layer
    W2 = tf.Variable(initial_value=tf.truncated_normal(
        shape=[16, 10], # Shape
        stddev=1/16**0.5, # Calibrating variance
        seed=0
    ))
    b2 = tf.Variable(initial_value=tf.zeros(shape=[10]))

    # Compute logits
    hidden = tf.nn.relu( # ReLU
        tf.matmul(X, W1) + b1)
    logits = tf.matmul(hidden, W2) + b2