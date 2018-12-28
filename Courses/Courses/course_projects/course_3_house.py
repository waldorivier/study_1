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
import math
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
    
    def __init__(self):
        return None
        
    def analyze_cols(self, df, results):
        _df = df.copy()

        null_values_max = _df.isnull().sum().max()
        if null_values_max > 0:
            threshes = np.arange(0, null_values_max, 100)
            
            def drop_cols(df, thresh):
                _df = df.dropna(thresh=thresh, axis=1)

                row = {}
                row['thresh'] = thresh
                row['shape'] = _df.shape[1]
                results.append(row)
        
            for thresh in threshes :
                drop_cols(_df, thresh)
    
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

#------------------------------------------------------------------------------
# class which implements utilities to access house price meta data (columns, 
# column's type, ordinal column code mapping, tansfomation of entire ordinal 
# columns of a dataframe
#------------------------------------------------------------------------------
class meta_data:
    _working_dir = None
    _df_meta = None

    def __init__(self, working_dir):
        self._working_dir = working_dir
        self._load_meta_data()

    def _load_meta_data(self):
        data_file = os.path.join(self._working_dir, 'meta_data.txt')

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
    
    def _map_ordinal_col(self, df, col):
       self._map_col(df, col, 'Ordinal')
       
    # all ordinal columns's code will be associated with numerical values  as defined in meta data
    def map_ordinal_cols(self, df):
        cols = self.get_cols_of_type('Ordinal')
        for col in cols:
            self._map_ordinal_col(df, col)

#------------------------------------------------------------------------------
# class which implements utilities to manage house price data
#------------------------------------------------------------------------------
class sample_data:
    _meta_data = None
    _working_dir = None
    _target = None
 
    _df_test_data = None
    _df_train_data = None
    _df_train_data_orig = None
 
    def __init__(self, working_dir, target, meta_data):
        assert working_dir is not None and len(working_dir) > 0
        assert target is not None and len(target) > 0
        assert meta_data is not None
        
        self._working_dir = working_dir
        self._meta_data = meta_data
        self._target = target

    def load_data(self):
        self._load_train_data()
        self._load_test_data()
        
    def prepare_train_data(self):
        assert self._df_train_data_orig is not None 

        # drop nan columns which contains null values     
        self._df_train_data = self._df_train_data_orig.dropna(axis=1).copy()

        # takes the log the target
        self._df_train_data[self._target] = np.log(self._df_train_data[self._target])
        self._meta_data.map_ordinal_cols(self._df_train_data)

    def prepare_test_data(self):
        assert self._df_test_data is not None

        self._meta_data.map_ordinal_cols(self._df_test_data)
        
    def get_cols_of_type(self, df, type):
        cols = df.columns.intersection(self._meta_data.get_cols_of_type(type))
        return cols
 
    def get_cols_wo_target(self):
        cols = self._df_train_data.columns.copy()
        cols = cols.drop(self._target)
        cols = cols.drop(['Order', 'PID'])
        return cols

    def _load_train_data(self):
        data_file = os.path.join(self._working_dir, 'house-prices.csv')
 
        # save a copy of original train data     
        self._df_train_data_orig = pd.read_csv(data_file)
 
    # columns which were removed 
    def get_removed_cols(self): 
        return self._df_train_data_orig.columns.difference(set(self._df_train_data.columns))

    def _load_test_data(self):
        data_file = os.path.join(self._working_dir, 'house-prices-test.csv')
        self._df_test_data = pd.read_csv(data_file)

    def analyze(self):
        ana = analyze()
        results = []
        ana.analyze_cols(self._df_train_data_orig, results)
        return pd.DataFrame(results)
 
    class result:
        comb_cols = None
        cols = None
        lr = None
        train_score = None
        test_score = None
        test_baseline  = None
        PID_pred = None
        y_te = None
        y_te_pred = None

        def as_dict(self):
            return {'PID_test' : self.PID_pred,
                    'comb_cols' : self.comb_cols,
                    'cols' : self.cols,
                    'lr' : self.lr,
                    'train_score' : self.train_score,
                    'test_score' : self.test_score,
                    'test_baseline' : self.test_baseline,
                    'y_te' : self.y_te,
                    'y_te_pred' : self.y_te_pred}
                
#---------------------------------------------------------------------------
def anp(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

pd.set_option('display.max_columns', 90)

#------------------------------------------------------------------------------
# Evaluates a model given a subset of colums choosen among those of
# the original data set 
#------------------------------------------------------------------------------
def perform_train(sample_data:sample_data, comb_cols, results):
    try:
        df = sample_data._df_train_data.copy()
        y = df[sample_data._target]
        df = df[comb_cols]
        
        cols = sample_data.get_cols_of_type(df, 'Nominal')
        if len(cols) > 0:
            df = pd.get_dummies(df, columns=cols)

        #----------------------------------    
        # TODO : remove eventual outliers 
        
        _X = df.values
        X_tr, X_te, y_tr, y_te = train_test_split(
           _X, y.values, train_size = 0.5, test_size = 0.5, random_state=1)

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

        r = sample_data.result()
        r.comb_cols = comb_cols
        r.cols = df.columns
        r.lr = lr
        r.train_score = np.sqrt(mse(y_pred_tr, y_tr))
        r.test_score = np.sqrt(mse(y_pred_te, y_te))
        r.test_baseline = np.sqrt(mse(y_pred_base, y_te))
        r.PID_test = None
        r.y_te = y_te
        r.y_te_pred = y_pred_te
        results.append(r)

    except:
        print(comb_cols)

#------------------------------------------------------------------------------
# Based on an optimal solution, predicts values from a given test sample 
#------------------------------------------------------------------------------
def perform_test(sample_data, optimal_train, results):
    assert sample_data is not None
    assert optimal_train is not None
    assert results is not None

    try:
        id = 'PID'
        df = sample_data._df_test_data.copy()
        PID = df[id]

        df = df[optimal_train.comb_cols]
        
        cols = sample_data.get_cols_of_type(df, 'Nominal')
        if len(cols) > 0:
            df = pd.get_dummies(df, columns=cols)

        # re-index test set in order to be compatible with train set 
        df = df.reindex(columns=optimal_result.cols)
        df.fillna(0, inplace=True)

        X = df.values
        y_pred_te = optimal_train.lr.predict(X)
  
        r = sample_data.result()
        r.PID_test = PID
        r.comb_cols = optimal_train.comb_cols
        r.cols = df.columns

        # transform 
        r.y_te_pred = np.exp(y_pred_te)
        results.append(r)

    except:
        print(optimal_train.comb_cols)

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
class model_selector:
    _prediction_file_name = 'house-prices-pred.csv'
    _sample_data = None
    _train_results = None
    _prediction = None

    def __init__(self, sample_data):
        assert sample_data is not None

        self._sample_data = sample_data
 
    #--------------------------------------------------------------------------
    # enumerates all possible combinations with two features 
    #--------------------------------------------------------------------------
    def run_combinations(self):
        self._train_results = []
        self._prediction = []
  
        self._sample_data.prepare_train_data()
        
        combinations = [list(x) for x in itertools.combinations(sample_data.get_cols_wo_target(), 2)]
        combinations = [random.choice(combinations) for i in np.arange(10)]
        for combination in combinations:
            perform_train(sample_data, combination, self._train_results)

        sample_data.prepare_pred_data()
        perform_test(sample_data, self._find_optimal_train(), self._prediction)
     
        self._write_result()
  
    #--------------------------------------------------------------------------
    # find an optimal test score among all train's run
    #--------------------------------------------------------------------------
    def _find_optimal_train(self):
        assert self._train_results is not None and len(self._train_results) > 0

        df = pd.DataFrame([x.as_dict() for x in self._train_results])
        i_opt = df['test_score'].idxmin()
        return df.iloc[i_opt,:]

    #--------------------------------------------------------------------------
    # write prediction 
    #--------------------------------------------------------------------------
    def _write_result(self):
        data_file = os.path.join(self._sample_data._working_dir, self._prediction_file_name)

        df = pd.concat([pd.DataFrame(self._prediction[0].PID_test), 
                        pd.DataFrame(self._prediction[0].y_te_pred)], axis=1)

        df.columns = ['PID', 'SalePrice']
        df.to_csv(data_file, index=False)

    #--------------------------------------------------------------------------
    # plot optimal trains 
    #--------------------------------------------------------------------------
    def _plot_optimal_train(self):
        opt = self._find_optimal_train()

        values = [opt.test_baseline,  
                  opt.train_score,
                  opt.test_score, ]
  
        fig, ax = plt.subplots()

        bar_width = 0.6
        index = np.arange(len(values))
        rects1 = ax.bar(index, values, bar_width, color='g')
        ax.set_xlabel('metrics')
        ax.set_ylabel('mse score')
        ax.set_title('score by metrics')
        ax.set_xticks(index)
        ax.set_xticklabels(('baseline test score', 'train score', 'test score'))
        ax.legend()

        fig.tight_layout()
        plt.show()
        
#------------------------------------------------------------------------------

working_dir = os.getcwd()
working_dir = os.path.join(working_dir,'course_projects', 'data', 'module_3')

meta_data = meta_data(working_dir)
sample_data = sample_data(working_dir, 'SalePrice', meta_data)
sample_data.load_data()

model_selector = model_selector(sample_data)
model_selector.run_combinations()



