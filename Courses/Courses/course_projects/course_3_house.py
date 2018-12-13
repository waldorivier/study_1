#-------------------------------------------------------------------------
# Module 3 : course project - predicting house prices
#-------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from scipy.linalg import lstsq
from scipy import stats

from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

import itertools
import utils

#------------------------------------------------------------------------------

pd.set_option('display.max_columns', 90)

working_dir = os.getcwd()
data_file = os.path.join(working_dir, 'course_projects', 'data', 'module_3', 'house-prices.csv')
df_origin = pd.read_csv(data_file)

#------------------------------------------------------------------------------

analyze = utils.analyze()
meta_data = utils.meta_data(working_dir)

results = []
analyze.analyze_columns(df_origin, results)

#------------------------------------------------------------------------------
# dropping all NAN values reduced the number of columns from 82 to 55

df_results = pd.DataFrame(results)

#------------------------------------------------------------------------------
# keep the features being removed

all_columns = set(df_origin.columns)
df_w = df_origin.dropna(axis=1).copy()

removed_columns = all_columns.difference(set(df_w.columns))
df_rm = df_origin[list(removed_columns)]

df_rm

#------------------------------------------------------------------------------
# change the dimentsion of the target sale price

target = 'SalePrice'

df_w[target] = np.log(df_w[target])

#------------------------------------------------------------------------------

meta_data.load_meta_data()