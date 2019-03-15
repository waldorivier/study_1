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
from sklearn.linear_model import LogisticRegression

# Create the estimator
from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

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

from sklearn import datasets

#------------------------------------------------------------------------------
pd.set_option('display.max_columns', 90)

file_name = 'chicago-crimes.csv'

working_dir = os.getcwd()
working_dir = os.path.join(working_dir,'course_projects', 'module_4')

#------------------------------------------------------------------------------

data_file = os.path.join(working_dir, file_name)
df_data = pd.read_csv(data_file)

from sklearn.cluster import KMeans

# Create k-means object
kmeans = KMeans(
    n_clusters=8,
    random_state=0, 
    n_jobs=-1
)

df_data.set_index(keys=['Case Number'], inplace=True)
features = ['Longitude', 'Latitude']

df_data_f = df_data[features]

X = df_data_f.values

kmeans.fit(X)
pd.Series(kmeans.labels_).value_counts()
kmeans.cluster_centers_

kmeans.score(X)

for k in np.arange(0, 7):
    df_cluster = df_data[kmeans.labels_ == k]
    print (k, df_cluster.groupby('Year')['Block'].count().mean())


for cluster in np.arange(8):
    # Get points in this cluster
    idx = (kmeans.labels_ == cluster)

    # Plot points
    plt.scatter(
        X[idx, 0], 
        X[idx, 1], 
        label='cluster {}'.format(cluster)
    )

    # Plot centroid
    centroid = kmeans.cluster_centers_[cluster]
    plt.plot(centroid[0], centroid[1], marker='*', color='black', markersize=18)

# Add legend and labels
plt.legend()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

#------------------------------------------------------------------------------
# PEPPER
#------------------------------------------------------------------------------
file_name = 'pepper.jpg'

data_file = os.path.join(working_dir, file_name)
img = Image.open(data_file)

a_img = np.array(img)
a_original_shape = a_img.shape

X = a_img.reshape(-1, 3)

#------------------------------------------------------------------------------
kmeans.n_clusters = 1
kmeans.fit(X)
pd.Series(kmeans.labels_).value_counts()

kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(int)

X_ = kmeans.cluster_centers_[kmeans.labels_]

a_img_reduced = X_.reshape(a_original_shape)
plt.imshow(a_img_reduced)
plt.show()

#------------------------------------------------------------------------------