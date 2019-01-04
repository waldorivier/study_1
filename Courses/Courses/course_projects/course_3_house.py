#-------------------------------------------------------------------------
# Module 3 : course project - predicting house prices
#-------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# columns of a dataframe, etc...)
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

    #------------------------------------------------------------------------------
    # return dictionnary build on code and code designation (used in special case
    # for MS SubClass feature) 
    #------------------------------------------------------------------------------
    def get_dict_nominal(self, col):
        dict_nominal = None
        if self._get_type_of_col(col) == 'Nominal':
            df_nominal = self._df_meta[self._df_meta['column'] == col]
            codes = df_nominal.code.values
            try:
                codes = codes.astype(int)
            except:
                codes = [i.strip() for i in df_nominal.code.values]
         
            dict_nominal = dict(zip(codes, df_nominal.code_designation.values))
        return dict_nominal

    def _map_col(self, df, col, type):
        if df.columns.contains(col):
            if self._get_type_of_col(col) == type:
                dict_ordinal = self.get_dict_ordinal(col)
                df[col] = df[col].map(dict_ordinal)
    
    def _map_ordinal_col(self, df, col):
       self._map_col(df, col, 'Ordinal')
       
    # all ordinal columns's code will be associated with numerical values as defined in meta data
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
        
    #------------------------------------------------------------------------------
    # 
    #------------------------------------------------------------------------------
    def prepare_train_data(self):
        assert self._df_train_data_orig is not None 

        # drop nan columns which contains null values     
        self._df_train_data = self._df_train_data_orig.dropna(thresh=2429, axis=1).copy()
        self._df_train_data.fillna(0, inplace=True)

        # takes the log the target
        self._df_train_data[self._target] = np.log(self._df_train_data[self._target])
        self._meta_data.map_ordinal_cols(self._df_train_data)

        # special case MS SubClass

        col = 'MS SubClass'
        dict_nominal = self._meta_data.get_dict_nominal(col)
        self._df_train_data[col] = self._df_train_data[col].map(dict_nominal)
    
    # columns which were removed in prepare step above
    def get_removed_cols(self): 
        return self._df_train_data_orig.columns.difference(set(self._df_train_data.columns))

    def prepare_prediction_data(self):
        assert self._df_test_data is not None

        self._meta_data.map_ordinal_cols(self._df_test_data)
        
    def get_cols_of_type(self, df, type):
        cols = df.columns.intersection(self._meta_data.get_cols_of_type(type))
        return cols
 
    def get_cols_wo_target(self):
        assert self._df_train_data is not None

        cols = self._df_train_data.columns.copy()
        cols = cols.drop(self._target)
        cols = cols.drop(['Order', 'PID'])
        return cols

    def _load_train_data(self):
        data_file = os.path.join(self._working_dir, 'house-prices.csv')
 
        # save a copy of original train data     
        self._df_train_data_orig = pd.read_csv(data_file)
 
    def _load_test_data(self):
        data_file = os.path.join(self._working_dir, 'house-prices-test.csv')
        self._df_test_data = pd.read_csv(data_file)

    def analyze(self):
        ana = analyze()
        results = []
        ana.analyze_cols(self._df_train_data_orig, results)
        return pd.DataFrame(results)
 
    #------------------------------------------------------------------------------
    # iterate over all features (eventually encoded) and plot SalePrice as response
    #------------------------------------------------------------------------------
    def iterate_pair_plot(self):
        col_cnt = len(self.get_cols_wo_target())

        for i in np.arange(0, col_cnt):
            self._pair_plot(i)
   
    def pair_plot(self, col):
        assert self._df_train_data.columns.contains(col)

        df = self._df_train_data.copy()
        df = df[[self._target, col]]
       
        if self._meta_data._get_type_of_col(col) == 'Nominal':
            df = pd.get_dummies(df, col)    

        cols = df.columns.drop(self._target)
        g = sns.pairplot(df, x_vars=cols,
            y_vars=[self._target], 
            kind="reg",
            plot_kws={'line_kws':{'color':'red'}})
        plt.show()

    def hist_plot(self, col):
        assert self._df_train_data.columns.contains(col)

        df = self._df_train_data.copy()
        df = df[col]
      
        if self._meta_data._get_type_of_col(col) == 'Nominal':
            df = pd.get_dummies(df, col)    

        df.hist()
        plt.show()

    def get_train_distribution(self):
        assert self._df_train_data is not None

        return np.exp(self._df_train_data.SalePrice).describe()
    
    def _indicator(self, col):
        def _f_fire_places(x):
            if x > 1:
                return 1
            else:
                return 0
            return _f

        def _f_year_built(x):
            if x > 1970:
                return 1
            else:
                return 0

        if col == 'Fireplaces':
            return _f_fire_places
        elif col == 'Year Built':
            return _f_year_built
        
    def apply_transformation(self, col):
        assert self._df_train_data.columns.contains(col)

        indicator = self._indicator(col)
        assert indicator is not None
        self._df_train_data[col] = self._df_train_data[col].apply(indicator)

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
# evaluates a model given a subset of colums choosen among those of
# the original data set 
#------------------------------------------------------------------------------
def run_train(reg_type, sample_data, comb_cols, results):
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

        if reg_type == 'linear':
            reg = LinearRegression()
        elif reg_type == 'huber':
            reg = HuberRegressor(1.35)
            y_tr = y_tr.flatten()
        elif reg_type == 'ridge':    
            reg = Ridge(0.04)

        reg.fit(X_tr, y_tr)
       
        #----------------------------------
        # negatives values not allowed 
     
        y_pred_tr = np.maximum(reg.predict(X_tr), 0)
        y_pred_te = np.maximum(reg.predict(X_te), 0)
   
        #----------------------------------
        # determine the baseline 
                
        dummy = DummyRegressor(strategy='mean')
        dummy.fit(X_tr, y_tr)
        y_pred_base = dummy.predict(X_te)   

        r = sample_data.result()
        r.comb_cols = comb_cols
        r.cols = df.columns
        r.lr = reg
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
# based on an optimal solution, return prediciton values for a given sample 
#------------------------------------------------------------------------------
def build_prediction(optimal_train, sample_data):
    assert sample_data is not None
    assert optimal_train is not None

    try:
        id = 'PID'
        df = sample_data._df_test_data.copy()
        PID = df[id]

        df = df[optimal_train.comb_cols]
        
        cols = sample_data.get_cols_of_type(df, 'Nominal')
        if len(cols) > 0:
            df = pd.get_dummies(df, columns=cols)

        # re-index test set in order to be compatible with train set 
        df = df.reindex(columns=optimal_train.cols)
        df.fillna(0, inplace=True)

        X = df.values
        y_pred_te = optimal_train.lr.predict(X)
  
        prediction = sample_data.result()
        prediction.PID_test = PID
        prediction.comb_cols = optimal_train.comb_cols
        prediction.cols = df.columns

        # transform 
        prediction.y_te_pred = np.exp(y_pred_te)

        # format 
        df = pd.concat([pd.DataFrame(prediction.PID_test), 
                        pd.DataFrame(prediction.y_te_pred)], axis=1)
        df.columns = ['PID', 'SalePrice']
        return df
        
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
 
    def reset(self):
        self._train_results = None
        self._prediction = None

    #--------------------------------------------------------------------------
    # enumerates all possible combinations with k features
    # perform and write perdictions
    # limit : max combinations 
    # reg_type : linear, huber
    #--------------------------------------------------------------------------
    def run_combinations(self, reg_type, k, limit):
        self._sample_data.prepare_train_data()

        cols = self._sample_data.get_cols_wo_target()
        cols_cnt = len(cols)

        # avoid too many combinations to be evaluated

        if (anp(cols_cnt, k) < 100000) :
            combinations = [list(x) for x in itertools.combinations(cols, k)]
            combinations = [random.choice(combinations) for i in np.arange(limit)]

            self._train_results = []
            for combination in combinations:
                run_train(reg_type, sample_data, combination, self._train_results)

            sample_data.prepare_prediction_data()
            self._prediction = build_prediction(self._find_optimal_train(), sample_data)
            self._write_prediction()
  
    #--------------------------------------------------------------------------
    # add columes to an (eventually already evaluated) optimal combination in order
    # to build a more accurate  model 
    # a new optimal train will be calculated
    #--------------------------------------------------------------------------
    def run_combination(self, reg_type, cols):
        self._sample_data.prepare_train_data()
        # when no optimal train already exists
        try :
            opt_train = self._find_optimal_train()
            combination = opt_train.comb_cols.copy()
        except:
            combination = []
        combination.extend(cols)

        self._train_results = []
        run_train(reg_type, sample_data, combination, self._train_results)

        sample_data.prepare_prediction_data()
        self._prediction = build_prediction(self._find_optimal_train(), sample_data)
   
    #--------------------------------------------------------------------------
    # find an optimal test score among all train's run
    #--------------------------------------------------------------------------
    def _find_optimal_train(self):
        assert len(self._train_results) > 0

        df = pd.DataFrame([x.as_dict() for x in self._train_results])
        i_opt = df['test_score'].idxmin()
        return df.iloc[i_opt,:]

    #--------------------------------------------------------------------------
    # get Saleprice distibution of the optimal train 
    #--------------------------------------------------------------------------
    def get_prediction_distribution(self):
        assert self._prediction is not None

        return self._prediction.SalePrice.describe()

    #--------------------------------------------------------------------------
    # write prediction 
    #--------------------------------------------------------------------------
    def _write_prediction(self):
        assert self._prediction is not None

        data_file = os.path.join(self._sample_data._working_dir, self._prediction_file_name)
        self._prediction.to_csv(data_file, index=False)
  
    #--------------------------------------------------------------------------
    # plot optimal train
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
        ax.set_title(opt.comb_cols)
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

model_selector.run_combinations('linear', 1, 2000)

optimal_cols =  ['Overall Qual', 'Gr Liv Area']

model_selector.reset()
model_selector.run_combination('linear', optimal_cols)
model_selector._find_optimal_train()
model_selector.get_prediction_distribution()

sample_data.apply_transformation('Fireplaces')
sample_data.apply_transformation('Year Built')

cols_to_add = ['Fireplaces', 'Lot Area', 'Year Built', 'TotRms AbvGrd', '1st Flr SF', 'MS SubClass', 'Central Air']

for col in cols_to_add:
    model_selector.run_combination('linear', [col])
    model_selector._find_optimal_train()
    model_selector.get_prediction_distribution()
    # model_selector._plot_optimal_train()

sample_data.get_train_distribution()

