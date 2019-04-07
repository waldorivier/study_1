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

# resolution = 3072
resolution = (784)

# file_name = 'mnist-6k.npz'
file_name = 'mnist-20k.npz'
# file_name = 'cifar10-6k.npz'
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

# Split again into validation/test sets
X_valid, X_test, y_valid, y_test = train_test_split(
    X_test, y_test,
    test_size=500, random_state=0
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_rescaled = scaler.fit_transform(X_train)
X_valid_rescaled = scaler.transform(X_valid)

#------------------------------------------------------------------------------

# Create a new graph
graph = tf.Graph()

if 0:
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

#------------------------------------------------------------------------------
# Batches with shuffle (mélange)
#------------------------------------------------------------------------------

def get_batches(X, y, batch_size):
    # Shuffle X,y
    shuffled_idx = np.arange(len(y)) # 1,2,...,n
    np.random.shuffle(shuffled_idx)

    # Enumerate indexes by steps of batch_size
    # i: 0, b, 2b, 3b, 4b, .. where b is the batch size
    for i in range(0, len(y), batch_size):
        # Batch indexes
        batch_idx = shuffled_idx[i:i+batch_size]
        yield X[batch_idx], y[batch_idx]

#------------------------------------------------------------------------------

with graph.as_default():
    # Create placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.int32, shape=[None])

    # Hidden layer with 16 units
    hidden = tf.layers.dense(
        X, 16, activation=tf.nn.relu, # ReLU
        kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=0),
        bias_initializer=tf.zeros_initializer(),
        name='hidden'
    )

    # Output layer
    logits = tf.layers.dense(
        hidden, 10, activation=None, # No activation function
        kernel_initializer=tf.variance_scaling_initializer(scale=1, seed=0),
        bias_initializer=tf.zeros_initializer(),
        name='output'
    )

    #------------------------------------------------------------------------------

    # Loss fuction: mean cross-entropy
    mean_ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))

    # Gradient descent
    lr = tf.placeholder(dtype=tf.float32)
    gd = tf.train.GradientDescentOptimizer(learning_rate=lr)

    # Minimize cross-entropy
    train_op = gd.minimize(mean_ce)

    # Compute predictions and accuracy
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    is_correct = tf.equal(y, predictions)
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))

# Train and validation accuracy after each epoch
train_acc_values = []
valid_acc_values = []

with tf.Session(graph=graph) as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Set seed
    np.random.seed(0)

    # Train several epochs
    for epoch in range(50):
        # Accuracy values (train) after each batch
        batch_acc = []

        # Get batches of data
        for X_batch, y_batch in get_batches(X_train_rescaled, y_train, 64):
            # Run training and evaluate accuracy
            _, acc_value = sess.run([train_op, accuracy], feed_dict={
                X: X_batch,
                y: y_batch,
                lr: 0.01 # Learning rate
            })

            # Save accuracy (current batch)
            batch_acc.append(acc_value)

        # Evaluate validation accuracy
        valid_acc = sess.run(accuracy, feed_dict={
            X: X_valid_rescaled,
            y: y_valid
        })
        valid_acc_values.append(valid_acc)

        # Also save train accuracy (we will use the mean batch score)
        train_acc_values.append(np.mean(batch_acc))

        # Print progress
        print('Epoch {} - valid: {:.3f} train: {:.3f} (mean)'.format(
            epoch+1, valid_acc, np.mean(batch_acc)
        ))

    # Weights of the hidden and output layers
    weights_hidden = graph.get_tensor_by_name('hidden/kernel:0').eval()
    weights_output = graph.get_tensor_by_name('output/kernel:0').eval()

plt.plot(train_acc_values, label='train')
plt.plot(valid_acc_values, label='valid')
plt.title('Validation accuracy {:.3f} (mean last 3)'.format(
    np.mean(valid_acc_values[-3:]) # last three values
))
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))

# Plot the weights of the 16 hidden units
for i, axis in enumerate(axes.flatten()):
    # Get weights of i-th hidden unit
    weights = weights_hidden[:, i]

    # Reshape into 28 by 28 array
    weights = weights.reshape(28, 28)

    # Plot weights
    axis.set_title('unit {}'.format(i+1))
    axis.imshow(weights, cmap=plt.cm.gray_r) # Grayscale
    axis.get_xaxis().set_visible(False) # Disable x-axis
    axis.get_yaxis().set_visible(False) # Disable y-axis

plt.show()