import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

import itertools
import math
import random as r  

#-------------------------------------------------------------------------------

dir = os.getcwd()
dir = os.path.join(dir,'course_projects', 'module_4')

data_df = pd.read_csv(os.path.join(dir, 'titanic.csv'))

#-------------------------------------------------------------------------------

encoded_df = pd.get_dummies(data_df, columns=['pclass'])
encoded_df.sex = [1 if (x=='male') else 0 for x in encoded_df.sex]

features = encoded_df.drop(['name', 'survived'], axis=1)

# Create X/y arrays
X = features.values
y = encoded_df.survived.values

dt = DecisionTreeClassifier(
    criterion='gini', max_depth=1, random_state=0)

dt.fit(X, y)

# Get score
dt.score(X, y)

dot_data = export_graphviz(
    dt, out_file=None,
    feature_names=features.columns, class_names=['died', 'survived'],
    filled=True, rounded=True, proportion=True   
)

graphviz.Source(dot_data)