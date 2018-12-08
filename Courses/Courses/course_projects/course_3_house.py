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
data_file = os.path.join(working_dir, 'course_projects', 'Data', 'module_3', 'house-prices.csv')
data_df = pd.read_csv(data_file)

analyze = utils.analyze()

results = []
analyze.analyze_columns(data_df, results)

df_results = pd.DataFrame(results)

