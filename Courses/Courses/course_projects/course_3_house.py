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

    def _get_type_of_col(self, col):
        try:
            return self._df_meta[self._df_meta['column'] == col]['type'].iloc[0]
        except:
            print ("Column not found in meta data " + col)
    
    def get_cols_of_type(self, type):
        cols = self._df_meta.copy()
        cols = cols[cols.type==type]['column'].drop_duplicates()
        return cols

    def map_ordinal_col(self, df, col):
       self._map_col(df, col, 'Ordinal')

    def get_dict_ordinal(self, col):
        dict_ordinal = None
        if self._get_type_of_col(col) == 'Ordinal':
            df_ordinal = self._df_meta[self._df_meta['column'] == col]
            codes = df_ordinal.code.values
            try:
                codes = codes.astype(float)
            except:
                codes = [i.strip() for i in df_ordinal.code.values]
         
            dict_ordinal = dict(zip(codes, df_ordinal.ordinal_value.values))
        return dict_ordinal

    def _map_col(self, df, col, type):
        if df.columns.contains(col):
            if self._get_type_of_col(col) == type:
                dict_ordinal = self.get_dict_ordinal(col)
                df[col] = df[col].map(dict_ordinal)

    # all ordinal columns's code will be associated with numerical values  
    # as defined in meta data
    def map_ordinal_cols(self, df):
       cols = self.get_cols_of_type('Ordinal')

       for col in cols:
           self.map_ordinal_col(df, col)

class sample_data:
    _meta_data = None
    _working_dir = None
    _df_target = None
 
    _df_test_data = None
    _df_train_data = None
    _df_train_data_orig = None
 
    def __init__(self, working_dir, meta_data, target):
        assert meta_data is not None
        
        self._working_dir = working_dir

        self._load_train_data()
        self._load_test_data()

        self._target = target
        self._meta_data = meta_data

    def prepare_train_data(self):
       # drop nan columns which contains null values     
        self._df_train_data = self._df_train_data_orig.dropna(axis=1).copy()

        # takes the log the target
        self._df_train_data[self._target] = np.log(self._df_train_data[self._target])

        self._meta_data.map_ordinal_cols(self._df_train_data)
        
    # columns which were removed after prepare above
    def get_removed_cols(self): 
        return self._df_train_data_orig.columns.difference(set(self._df_train_data.columns))
    
    def get_cols_of_type(self, type):
        cols = self._df_train_data.columns.intersection(self._meta_data.get_cols_of_type(type))
        return cols
 
    def get_cols_wo_target(self):
        cols = self._df_train_data.columns.copy()
        cols = cols.drop(self._target)
        cols = cols.drop(['Order', 'PID'])
        return cols

    def _load_train_data(self):
        data_file = os.path.join(self._working_dir, 'course_projects', 'data', 'module_3', 'house-prices.csv')
 
        # save a copy of original train data     
        self._df_train_data_orig = pd.read_csv(data_file)
 
    def _load_test_data(self):
        data_file = os.path.join(self._working_dir, 'course_projects', 'data', 'module_3', 'house-prices-test.csv')
        self._df_test_data = pd.read_csv(data_file)
 
    class result:
        _comb_cols      = None
        _cols           = None
        _coef           = None
        _train_score    = None
        _test_score     = None
        _test_baseline  = None
        _y_te           = None
        _y_te_pred      = None

        def as_dict(self):
            return {'comb_cols'      : self._ccomb_cols,
                    'cola'           : self._cols,
                    'coef'           : self._coef,
                    'train_score'    : self._train_score,
                    'test_score'     : self._test_score,
                    'test_baseline'  : self._test_baseline,
                    'y_te'           : self._y_te,
                    'y_te_pred'      : self._y_te_pred}
                
#---------------------------------------------------------------------------

pd.set_option('display.max_columns', 90)
working_dir = os.getcwd()

#------------------------------------------------------------------------------

if 0:
    analyze = analyze()
    results = []
    analyze.analyze_columns(df_origin, results)

    df_results = pd.DataFrame(results)

#------------------------------------------------------------------------------
# test a model given a subset of colums 
#------------------------------------------------------------------------------
def perform_test(sample_data:sample_data, comb_cols, results):
    try:
        _df = sample_data._df_train_data.copy()
    
        _y = _df[sample_data._target]
        _df = _df[comb_cols]
        
        cols = sample_data.get_cols_of_type('Nominal')
        if len(cols) > 0:
            _df = pd.get_dummies(_df, columns=cols)

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

        r = sample_data._result()
        r._cols_subset    = comb_cols
        r._cols          = _df.cols
        r._coef           = lr.coef_
        r._train_score    = np.sqrt(mse(y_pred_tr, y_tr))
        r._test_score     = np.sqrt(mse(y_pred_te, y_te))
        r._test_baseline  = np.sqrt(mse(y_pred_base, y_te))
        r._y_te           = pd.Series(y_te)
        r._y_te_pred      = pd.Series(y_pred_te)

        results.append(r)
    
    except:
        print(comb_cols)

#------------------------------------------------------------------------------

meta_data = meta_data(working_dir)
sample_data = sample_data(working_dir, meta_data, 'SalePrice')
sample_data.prepare_train_data()

results = []
combinations = [list(x) for x in itertools.combinations(sample_data.get_cols_wo_target(), 2)]
combinations = [random.choice(combinations) for i in np.arange(1000)]
for combination in combinations:
    perform_test(sample_data, combination, results)

df_results = pd.DataFrame([x.as_dict() for x in results])

i_min = df_results['test_score'].idxmin()
df_results.iloc[i_min,:]

i_min = df_results['train_score'].idxmin()
best_train = df_results.iloc[i_min,:]
best_train




