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
import random

class analyze:
    #------------------------------------------------------------------------------
    # analyze columns 
    #
    # applies series of threshes on the df and mesures the number
    # of remaining columns
    # 
    # plot the result  
    #------------------------------------------------------------------------------
    def analyze_columns(self, df, results):
        _df = df.copy()

        null_values_max = _df.isnull().sum().max()
        if null_values_max > 0:
            threshes = np.arange(0, null_values_max, 100)
            
            def select_columns(df, thresh):
                row = {}
                _df = df.dropna(thresh=thresh, axis=1)
                row['thresh'] = thresh
                row['shape'] = _df.shape[1]
                results.append(row)
        
            for thresh in threshes :
                select_columns(_df, thresh)
    
            df_results = pd.DataFrame(results)
            df_results.set_index('shape', inplace=True)
        
            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.suptitle("numbers of remaining columns in terms of thresh for a df of shape (" +  
                            str(df.shape[0]) + "," + str(df.shape[1]) + ")")
        
            x_pos = np.arange(len(df_results))
            ax.bar(x_pos, df_results.index, align='center', color='green')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_results['thresh'], rotation=45)

            ax.set_ylabel("remaining columns")
            ax.set_xlabel("thresh")

            plt.show()

class meta_data:
    _working_dir = ""
    _df_meta = None

    def __init__(self, working_dir):
        self._working_dir = working_dir
        self._load_meta_data()

    def _load_meta_data(self):
        data_file = os.path.join(self._working_dir, 'course_projects', 'data', 'module_3', 'meta_data.txt')

        self._df_meta = pd.DataFrame.from_csv(data_file, sep='\t')
        self._df_meta.reset_index(inplace=True)
        self._df_meta.ffill(inplace=True)

    def _get_type(self, column):
        try:
            return self._df_meta[self._df_meta.column == column]['type'].iloc[0]
        except:
            print ("Column not found in meta data " + column)
    
    def get_type_columns(self, type):
        columns = self._df_meta.copy()
        columns = columns[columns.type==type]['column'].drop_duplicates()
        return columns

    def transform_ordinal_column(self, df, column):
       self._transform_column(df, column, 'Ordinal')

    def get_dict_ordinal(self, column):
        dict_ordinal = None
        if self._get_type(column) == 'Ordinal':
            df_ordinal = self._df_meta[self._df_meta['column'] == column]
            codes = df_ordinal.code.values
            try:
                codes = codes.astype(float)
            except:
                codes = [i.strip() for i in df_ordinal.code.values]
         
            dict_ordinal = dict(zip(codes, df_ordinal.ordinal_value.values))
        return dict_ordinal

    def _transform_column(self, df, column, type):
        if df.columns.contains(column):
            if self._get_type(column) == type:
                dict_ordinal = self.get_dict_ordinal(column)
                df[column] = df[column].map(dict_ordinal)

    def transform_ordinal_df(self, df):
       ordinal_columns = self.get_type_columns('Ordinal')

       for column in ordinal_columns:
           self.transform_ordinal_column(df, column)

class custom_data:
    _meta_data = None
    _working_dir = None
    _df_test_data = None

    def __init__(self, working_dir, meta_data):
        self._working_dir = working_dir
        self._meta_data = meta_data
        
        self._load_test_data()

    def get_type_columns(self, df : pd.DataFrame, type):
        columns = df.columns.intersection(self._meta_data.get_type_columns(type))
        return columns

    def _load_test_data(self):
        data_file = os.path.join(self._working_dir, 'course_projects', 'data', 'module_3', 'house-prices-test.csv')
        self._df_test_data = pd.read_csv(data_file)

    
class result:
    _columns_subset = None
    _columns        = None
    _coef           = None
    _train_score    = None
    _test_score     = None
    _test_baseline  = None
    _y_te           = None
    _y_te_pred      = None

    def as_dict(self):
        return {'columns_subset' : _columns_subset,
                'columns'        : _columns,
                'coef'           : _coef,
                'train_score'    : _train_score,
                'test_score'     : test_score,
                'test_baseline'  : _test_baseline,
                'y_te'           : _y_te,
                'y_te_pred'      : _y_te_pred}
                
#---------------------------------------------------------------------------

pd.set_option('display.max_columns', 90)

working_dir = os.getcwd()
data_file = os.path.join(working_dir, 'course_projects', 'data', 'module_3', 'house-prices.csv')
df_origin = pd.read_csv(data_file)

#------------------------------------------------------------------------------

if 0:
    analyze = analyze()
    results = []
    analyze.analyze_columns(df_origin, results)

    df_results = pd.DataFrame(results)

#------------------------------------------------------------------------------
# keep the features being removed

removed_columns = all_columns.difference(set(df.columns))
df_rm = df_origin[list(removed_columns)]

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# test a model given a subset of colums 
#------------------------------------------------------------------------------
def perform_test(df:pd.DataFrame, custom_data:custom_data, columns_subset, target, results):

    try:
        _df = df.copy()
    
        _y = _df[target]
        _df = _df[columns_subset]
        
        nominal_columns = custom_data.get_type_columns(_df, 'Nominal')
        if len(nominal_columns) > 0:
            _df = pd.get_dummies(_df, columns=nominal_columns)

        #----------------------------------    
        # TODO : remove eventual outliers 
        
        _X = _df.values

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

        r = result()
        r._columns_subset = columns_subset
        r._columns        = _df.columns
        r._coef           = lr.coef_
        r._train_score    = np.sqrt(mse(y_pred_tr, y_tr))
        r._test_score     = np.sqrt(mse(y_pred_te, y_te))
        r._test_baseline  = np.sqrt(mse(y_pred_base, y_te))
        r._y_te           = pd.Series(y_te)
        r._y_te_pred      = pd.Series(y_pred_te)

        results.append(r)
    
    except:
        print (columns_subset)

#------------------------------------------------------------------------------

all_columns = set(df_origin.columns)
df = df_origin.dropna(axis=1).copy()

meta_data = meta_data(working_dir)
custom_data = custom_data(working_dir, meta_data)

target = 'SalePrice'
df[target] = np.log(df[target])

meta_data.transform_ordinal_df(df)

columns = df.columns.copy()
columns_wo_target = columns.drop(target)
columns_wo_target = columns_wo_target.drop(['Order', 'PID'])

results = []
combinations = [list(x) for x in itertools.combinations(columns_wo_target, 5)]
combinations = [random.choice(combinations) for i in np.arange(1000)]
for combination in combinations:
    perform_test(df, custom_data, combination, target, results)

p

i_min = df_results['test_score'].idxmin()
df_results.iloc[i_min,:]

i_min = df_results['train_score'].idxmin()
best_train = df_results.iloc[i_min,:]
best_train[1]




