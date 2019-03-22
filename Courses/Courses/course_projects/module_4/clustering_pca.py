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
from sklearn.cluster import KMeans

from sklearn.model_selection import GridSearchCV

from scipy.linalg import lstsq
from scipy import stats

from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

import itertools
import math
import random as r

import  PIL
from PIL import Image

from sklearn import datasets

#------------------------------------------------------------------------------
pd.set_option('display.max_columns', 90)

working_dir = os.getcwd()
working_dir = os.path.join(working_dir,'course_projects', 'module_4')

#------------------------------------------------------------------------------

if 0:
    file_name = 'chicago-crimes.csv'
    data_file = os.path.join(working_dir, file_name)
    df_data = pd.read_csv(data_file)


    # Create k-means object
    kmeans = KMeans(
        n_clusters=8,
        random_state=0 # Fix results
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

if 0:
    file_name = 'pepper.jpg'

    data_file = os.path.join(working_dir, file_name)
    img = Image.open(data_file)

    a_img = np.array(img)
    a_original_shape = a_img.shape

    X = a_img.reshape(-1, 3)

    #------------------------------------------------------------------------------
    kmeans.n_clusters = 5
    kmeans.fit(X)
    pd.Series(kmeans.labels_).value_counts()

    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(int)

    X_ = kmeans.cluster_centers_[kmeans.labels_]

    a_img_reduced = X_.reshape(a_original_shape)
    plt.imshow(a_img_reduced)
    plt.show()

#------------------------------------------------------------------------------
# PCA
#------------------------------------------------------------------------------

file_name = 'wine-data.csv'
data_file = os.path.join(working_dir, file_name)
df_data = pd.read_csv(data_file)

features = df_data.drop('kind', axis=1)
X = features.values
y = df_data.kind.values

print('X:', X.shape) # (178, 13)
print('y:', y.shape) # (178,)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plot_pca(X_2d)
print_result(X, features, pca)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca.fit_transform(X_scaled)

print_result(X_scaled, features, pca).iloc[:,1].sum()
print_result(X_scaled, features, pca).iloc[:,2].sum()

#------------------------------------------------------------------------------
# Pipeline version
#------------------------------------------------------------------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',PCA(n_components=2))
    ])

plot_pca(pipe.fit_transform(X))

pca_components_ = pipe.named_steps['pca'].components_
scaler_ = pipe.named_steps['scaler']


#------------------------------------------------------------------------------
def plot_pca(X):
    sns.set()

    # Plot each kind of wine
    for kind in [1, 2, 3]:
        # Wine samples of this type
        idx = (y == kind)

        # Plot their components
        plt.scatter(
            X[idx, 0], X[idx, 1],
            label='type {}'.format(kind)
        )

    # Labels and legend
    plt.legend()
    plt.xlabel('1st component')
    plt.ylabel('2nd component')
    plt.show()

#------------------------------------------------------------------------------
def print_result(X, features, pca):
    results_df = pd.DataFrame.from_items([
        ('variance', X.var(axis=0)),
        ('1st component', pca.components_[0]),
        ('2nd component', pca.components_[1])
        ]).set_index(features.columns)

    # Sort DataFrame by variance
    results_df.sort_values('variance', ascending=False, inplace=True)
    return results_df
