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

working_dir = os.getcwd()
working_dir = os.path.join(working_dir,'course_projects', 'module_4')
  
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
file_name = 'mnist-6k.npz'
data_file = os.path.join(working_dir, file_name)

with np.load(data_file, allow_pickle=False) as npz_file:
    data = npz_file['data']
    labels = npz_file['labels']

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
#-------------------------------------------------------------------------------

dummy.predict(X_te)

#-------------------------------------------------------------------------------
pipe = Pipeline([
    # ('scaler', StandardScaler()),
    # Create k-NN estimator without setting k
    ('knn', KNeighborsClassifier())
])

train_curve = []
test_curve = []

k_values = np.arange(1, 20, 1) 
for k in k_values:
    # Set k

    # parameter must be prefixed by pipe's step
    pipe.set_params(knn__n_neighbors=k)

    # Fit k-NN
    pipe.fit(X_tr, y_tr)

    # Compute train/test accuracy
    # train_acc = pipe.score(X_tr, y_tr)
    test_acc = pipe.score(X_te, y_te)

    # Save accuracy values
    # train_curve.append(train_acc)
    test_curve.append(test_acc)

# plt.plot(k_values, train_curve, label='train')
plt.plot(k_values, test_curve, label='test')
plt.legend()
plt.show()

#-------------------------------------------------------------------------------
# try to predict my own numnber
# convert image to matrix
#-------------------------------------------------------------------------------
a_img = format_number_image(0,256)
plt.imshow(a_img)
plt.show()

pipe.set_params(knn__n_neighbors=10)
pipe.fit(X_tr, y_tr)
pipe.predict(a_img.reshape(1,-1))



