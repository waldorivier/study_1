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
# change the dimension of the target 

target = 'SalePrice'
df_w[target] = np.log(df_w[target])

#------------------------------------------------------------------------------

meta_data.load_meta_data()
df_meta = meta_data._df_meta

#------------------------------------------------------------------------------
# test a model given a subset of colums 
#------------------------------------------------------------------------------
def perform_test(df, meta_data, columns_subset, target, results):

    _df = df.copy()
    
    _y = _df[target]
    _df = _df[columns_subset]
 
    # identifies categorical colums among subset in order to encod them
   
    categorical_columns = []
    for c in columns_subset:
        if meta_data.get_type(c) == 'Nominal':
            categorical_columns.append(c)

    _df = pd.get_dummies(_df, columns=categorical_columns)

    #----------------------------------    
    # TODO : remove eventual outliers 
        
    _X = _df.values

    #----------------------------------
    # TODO : if type is ordinal choose the ad-hoc encoding corresponding to
    # .....
  
    X_tr, X_te, y_tr, y_te = train_test_split(
       _X, _y.values, train_size = 0.5, test_size = 0.5, random_state=1)

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)

    #----------------------------------
    # negatives values not allowed 

    y_pred_tr = np.maximum(lr.predict(X_tr), 0)
    y_pred_te = np.maximum(lr.predict(X_te), 0)
   
    #----------------------------------
    # determine the baseline 

    dummy = DummyRegressor()
    dummy.fit(X_tr, y_tr)
    y_pred_base = dummy.predict(X_te)

    row = {}
    row['colums']       = columns_subset
    row['train_score']  = np.sqrt(mse(y_pred_tr, y_tr))
    row['test_score']   = np.sqrt(mse(y_pred_te, y_te))
    row['test-baseline']= np.sqrt(mse(y_pred_base, y_te))
    row['y_te']         = pd.Series(y_te)
    row['y_pred_te']    = pd.Series(y_pred_te)

    results.append(row)

#------------------------------------------------------------------------------

target = 'SalePrice'
columns_subset = ['Year Built', 'Land Contour']

columns = df_w.columns.copy()
columns_wo_target = columns.drop(target)
columns_wo_target = columns_wo_target.drop(['Order', 'PID'])

results = []
for i in np.arange(1, len(columns_wo_target) + 1):
    combinations = [list(x) for x in itertools.combinations(columns_wo_target, 2)]

    for combination in combinations:
        perform_test(df_w, meta_data, combination, target, results)

df_results = pd.DataFrame(results)
df_results['train_score']