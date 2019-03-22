import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

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
sns.set_palette(sns.color_palette("hls", 20))

working_dir = os.getcwd()
working_dir = os.path.join(working_dir,'course_projects', 'module_4')

#------------------------------------------------------------------------------
# PCA exercices
#------------------------------------------------------------------------------

file_name = 'mnist-10k.npz'
data_file = os.path.join(working_dir, file_name)

with np.load(data_file, allow_pickle=False) as npz_file:
    data = npz_file['data']
    labels = npz_file['labels']
    
X = data

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plot_pca(X_2d, labels)
print_result(X, np.arange(X.shape[1]), pca)

#------------------------------------------------------------------------------
def plot_pca(X, y):
    # sns.set()

    for kind in np.arange(2):
        # Wine samples of this type
        idx = (y == kind)

        # Plot their components
        plt.scatter(
            X[idx, 0], X[idx, 1],
            marker="${}$".format(kind)
            #label='type {}'.format(kind), 
        )

    # Labels and legend
    plt.legend()
    plt.xlabel('1st component')
    plt.ylabel('2nd component')
    plt.show()

#------------------------------------------------------------------------------
def print_result(X, i, pca):
    results_df = pd.DataFrame.from_items([
        ('variance', X.var(axis=0)),
        ('1st component', pca.components_[0]),
        ('2nd component', pca.components_[1])
        ]).set_index(i)

    # Sort DataFrame by variance
    # results_df.sort_values('variance', ascending=False, inplace=True)
    results_df.sort_values('1st component', ascending=False, inplace=True)
    return results_df

