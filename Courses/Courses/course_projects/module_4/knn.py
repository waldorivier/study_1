import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.dummy import DummyRegressor
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from scipy.linalg import lstsq
from scipy import stats

from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

import itertools
import math
import random as r

import  PIL
from PIL import Image

#------------------------------------------------------------------------------

file_name = 'heart-disease.csv'

working_dir = os.getcwd()
working_dir = os.path.join(working_dir,'course_projects', 'module_4')

#------------------------------------------------------------------------------

data_file = os.path.join(working_dir, file_name)
df_orig = pd.read_csv(data_file)

df = df_orig.copy()

X = df.drop('disease', axis=1).values
y = df.disease.values
 
X_tr, X_te, y_tr, y_te = train_test_split (
    X, y, test_size=0.3, random_state=0)

# most frequent baseline
np.sum(y_tr == 'absence') / len (y_tr)

pd.value_counts(y_tr) / len(y_tr)

# which correspond to the use of this classifier
dummy = DummyClassifier(strategy = 'most_frequent')

dummy.fit(X_tr, y_tr)
accuracy = dummy.score(X_te, y_te)

# standardize data set / pay attention that only using train set to fit scaler
scaler = StandardScaler()
scaler.fit(X_tr)

X_tr_stand = scaler.transform(X_tr)
X_te_stand = scaler.transform(X_te)

X_te_stand.mean(axis=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_tr_stand, y_tr)
knn.predict(X_te)

accuracy = knn.score(X_te_stand, y_te)

#------------------------------------------------------------------------------
# using pipelines

pipe = Pipeline([
    ('scaler', scaler),
    ('knn', knn)
])

pipe.fit(X_tr, y_tr)
accuracy = pipe.score(X_te, y_te)

#------------------------------------------------------------------------------
# GRID search

pipe = Pipeline([
    ('scaler', StandardScaler()),
    # Create k-NN estimator without setting k
    ('knn', KNeighborsClassifier())
])

train_curve = []
test_curve = []

k_values = np.arange(1, 100, 5) 
for k in k_values:
    # Set k

    # parameter must be prefixed by pipe's step

    pipe.set_params(knn__n_neighbors=k)

    # Fit k-NN
    pipe.fit(X_tr, y_tr)

    # Compute train/test accuracy
    train_acc = pipe.score(X_tr, y_tr)
    test_acc = pipe.score(X_te, y_te)

    # Save accuracy values
    train_curve.append(train_acc)
    test_curve.append(test_acc)

plt.plot(k_values, train_curve, label='train')
plt.plot(k_values, test_curve, label='test')
plt.legend()
plt.show()

#-------------------------------------------------------------------------------
# working with images

file_name = 'mnist-img.png'
data_file = os.path.join(working_dir, file_name)

img = Image.open(data_file)
a_img = np.array(img)

plt.imshow(a_img)
plt.show()

plt.imshow(a_img, cmap=plt.cm.gray_r)
plt.show()

file_name = 'cifar-img.png'
data_file = os.path.join(working_dir, file_name)

img = Image.open(data_file)
a_img = np.array(img)

plt.imshow(a_img[:,:,0])
plt.show()

#-------------------------------------------------------------------------------
# Festures Matrix and how to convert / store images in a Matrix fot further analysis

files = [
    'cifar-1.png',
    'cifar-2.png',
    'cifar-3.png',
    'cifar-4.png',
    'cifar-5.png'
]

# Feature matrix
features = []

for file_name in files:
    # Load the image
    data_file = os.path.join(working_dir, file_name)
    pillow_img = Image.open(data_file)

    # Convert it into a Numpy array
    img = np.array(pillow_img)

    # Flatten the array
    flat_img = img.flatten()

    # Add it to the feature matrix
    features.append(flat_img)

X = np.array(features)
X = X.astype(np.float)

#-------------------------------------------------------------------------------
# binary file format

import pickle

data = {
    'x': [6.28318, 2.71828, 1],
    'y': [2, 3, 5]
}

file_name = 'data.p'
data_file = os.path.join(working_dir, file_name)

# dump / w rite / b inary
with open(data_file, 'wb') as file:
    pickle.dump(data, file)

# reload
with open(data_file, 'rb') as file:
    data_ = pickle.load(file)

data = {
    'x': np.array([6.28318, 2.71828, 1]),
    'y': np.array([2, 3, 5])
}

file_name = 'data.npy'
data_file = os.path.join(working_dir, file_name)
np.save(data_file, data)

data = np.load(data_file)

#-------------------------------------------------------------------------------
x = np.array([6.28318, 2.71828, 1], dtype=np.float16)
y = np.array([2, 3, 5])

file_name = 'data.npz'
data_file = os.path.join(working_dir, file_name)
np.savez(data_file, features=x, targets=y)

with np.load(data_file, allow_pickle=False) as npz_file:
    # It's a dictionary-like object
    print(npz_file.keys())
    # Prints: ['features', 'targets']§

    # Load the arrays
    print('x:', npz_file['features'])
    # Prints: [6.28125, 2.71875, 1.]

    print('y:', npz_file['targets'])
    # Prints: [2, 3, 5]
    
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# original image

def format_number_image(min, max):
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

    # filter < thresh -> 0
    a_reduced_img = np.array(reduced_img)
    l_flat_img = a_reduced_img.flatten()

    l_filt_img = []
    for i in l_flat_img:
        if i <= max and i > min :
            l_filt_img.append(i + 50)
        else :
            l_filt_img.append(0)

    reduced_img = np.reshape(l_filt_img, (28,28))

    # save it 
    file_name = 'number_waldo.png'
    data_file = os.path.join(working_dir, file_name)

    np.save(file_name, reduced_img)
    
    return  np.array(reduced_img)

#-------------------------------------------------------------------------------

file_name = 'mnist-6k.npz'
data_file = os.path.join(working_dir, file_name)

with np.load(data_file, allow_pickle=False) as npz_file:
    data = npz_file['data']
    labels = npz_file['labels']

for i, d in enumerate(data):
    d = data[i,:]
    d = d.reshape(28,28)

    plt.imshow(d)
    plt.show()

    if i > 1:
        break

#-------------------------------------------------------------------------------
# train / test split 

X_tr, X_te, y_tr, y_te = train_test_split (
    data, labels, test_size=1/6, random_state=0)

#-------------------------------------------------------------------------------
# categories and proportion of each one

categories = pd.Series(y_tr)
l = len(categories)

categories = categories.value_counts()*100/l
df_categories = pd.DataFrame(categories, columns=['nb'])
plt.bar(df_categories.index, df_categories.nb)
plt.show()

rand_idx = r.randint(0,5000)
plt.imshow(X_tr[rand_idx].reshape((28,28)))
plt.show()

print (y_tr[rand_idx])

dummy = DummyClassifier(strategy = 'most_frequent')
dummy.fit(X_tr, y_tr)
accuracy = dummy.score(X_te, y_te)

#-------------------------------------------------------------------------------
# returns a array of one; in fact it appears that *most frequent" within y_tr
# is one 

dummy.predict(X_te)

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# try to predict my own numnber
# convert image to matri

a_img = format_number_image(125,140)
plt.imshow(a_img)
plt.show()

# transfomr an array to a matrix with one row
# a_img.reshape(1,-1)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_tr, y_tr)

knn.predict(X_te)
accuracy = knn.score(X_te, y_te)
knn.predict(a_img.reshape(1,-1))

pipe = Pipeline([
    # s('scaler', StandardScaler()),
    # Create k-NN estimator without setting k
    ('knn', KNeighborsClassifier())
])

train_curve = []
test_curve = []

k_values = np.arange(1, 10, 5) 
for k in k_values:
    # Set k

    # parameter must be prefixed by pipe's step

    pipe.set_params(knn__n_neighbors=k)

    # Fit k-NN
    pipe.fit(X_tr, y_tr)

    # Compute train/test accuracy
    train_acc = pipe.score(X_tr, y_tr)
    test_acc = pipe.score(X_te, y_te)

    # Save accuracy values
    train_curve.append(train_acc)
    test_curve.append(test_acc)

plt.plot(k_values, train_curve, label='train')
plt.plot(k_values, test_curve, label='test')
plt.legend()
plt.show()





