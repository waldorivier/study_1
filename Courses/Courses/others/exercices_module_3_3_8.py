import os
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import random
import googletrans as translator
from pandas.tseries.offsets import *
import calendar
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.dummy import DummyRegressor

#-------------------------------------------------------------------------
# répertoire de travail
#-------------------------------------------------------------------------

working_dir = PureWindowsPath(os.getcwd())
data_dir = PureWindowsPath(working_dir.joinpath('Data').joinpath('module_3'))
data_result_name = 'result.csv'

pd.set_option('display.max_columns', 30)

#-------------------------------------------------------------------------
# 3.3.8 Exercices
#-------------------------------------------------------------------------

def mae(y, y_pred):
    return np.mean(np.abs(y-y_pred))

def mse(y, y_pred):
    return np.mean(np.square(y - y_pred))

to_plot = False

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fit_dummy(x, y):
    params = []
    param = {'id_fit' : 0, 'coefs' : (), 'rmse' : 0, 'mae' : 0, 'typ' : "", 'fit_step' : ""}

    if to_plot:
        plt.scatter(x, y)

    dummy = DummyRegressor(strategy='median')
    dummy.fit(x[:, np.newaxis], y)

    y_pred = dummy.predict(x[:, np.newaxis])

    rmse = np.sqrt(mse(y, y_pred))
    res_mae = mae(y, y_pred)

    param['typ'] = 'dummy'
    param['fit_step'] = 'train'
    param['id_fit'] = 1
    param['coefs'] = ''
    param['rmse'] = rmse
    param['mae'] = res_mae

    params.append(param)

    if to_plot:
        plt.plot(x, y_pred, label = param)
        plt.show()

    return pd.DataFrame(params)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fit_huber(x, y):
    params = []
    if to_plot:
        plt.scatter(x, y)
    
    i = 0
    
    try:
        for ep in np.linspace(1.1, 1.5, 8):
            param = {'id_fit' : 0, 'coefs' : (), 'rmse' : 0, 'mae' : 0, 'typ' : "", 'fit_step' : ""}
            
            lr_huber = HuberRegressor(epsilon=ep)
            lr_huber.fit(x[:, np.newaxis], y)
            y_pred = lr_huber.predict(
                x[:, np.newaxis] 
            )

            coefs = (ep, lr_huber.coef_, lr_huber.intercept_)
              
            rmse = np.sqrt(mse(y, y_pred))
            res_mae = mae(y, y_pred)
            
            param['typ'] = 'huber'
            param['fit_step'] = 'train'
            param['id_fit'] = i
            param['coefs'] = coefs
            param['rmse'] = rmse
            param['mae'] = res_mae

            params.append(param)
            
            if to_plot:
                plt.plot(x, y_pred, 
                         color=sns.color_palette()[np.mod(i,6)], 
                         label=param)
                i = i + 1
        
        if to_plot:
            plt.legend()        
            plt.show()
 
    except ValueError :
        print (i)

    return pd.DataFrame(params)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fit_poly(x, y):
    params = []
    
    if to_plot:  
        plt.scatter(x, y)

    for i in (1, 3, 5):
        param = {'id_fit' : 0, 'coefs' : (), 'rmse' : 0, 'mae' : 0, 'typ' : "", 'fit_step' : ""}

        coefs = np.polyfit(x, y, deg = i)
        y_pred = np.polyval(coefs, x)

        rmse = np.sqrt(mse(y, y_pred))
        res_mae = mae(y, y_pred)

        param['typ'] = 'poly'
        param['fit_step'] = 'train'
        param['id_fit'] = i
        param['coefs'] = coefs
        param['rmse'] = rmse
        param['mae'] = res_mae

        params.append(param)
        
        if to_plot:
            plt.plot(x, y_pred, label=param, linewidth=1)

    if to_plot:            
        plt.legend()
        plt.show()

    return pd.DataFrame(params)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fit_compare():
    
    data_file = data_dir.joinpath('bike-sharing-train.csv')
    df_train = pd.read_csv(data_file)

    df_train.sort_values(by=['temp'], inplace=True)

    x_train = df_train.temp.values
    y_train = df_train.users.values

    df_poly = fit_poly (x_train, y_train)
    df_huber = fit_huber (x_train, y_train)
    df_dummy = fit_dummy (x_train, y_train)

    df_compare = pd.concat([df_poly, df_huber, df_dummy])
    df_compare.index = pd.RangeIndex(len(df_compare.index))

    return df_compare

#------------------------------------------------------------------------------
# Apply fit selection on test data
# 
# return  df_test
#------------------------------------------------------------------------------
def fit_test(df_fit):
    data_file =  data_dir.joinpath('bike-sharing-test.csv')
    df_test = pd.read_csv(data_file)

    df_test.sort_values(by=['temp'], inplace=True)
    x_test = df_test.temp.values
    y_test = df_test.users.values
   
    params = []
    for row in df_fit.itertuples():
        param = {'id_fit' : 0, 'coefs' : (), 'rmse' : 0, 'mae' : 0, 'typ' : "", 'fit_step' : ""}
        y_pred = None;

        if row.typ == 'poly':
            y_pred = np.polyval(row.coefs, x_test)
    
        if row.typ == 'huber':
            ep = row.coefs[0]
            coef = row.coefs[1]
            intercept = row.coefs[2]
        
            lr_huber = HuberRegressor(epsilon=ep)
            lr_huber.coef_ = coef
            lr_huber.intercept_ = intercept

            y_pred = lr_huber.predict(
                    x_test[:, np.newaxis] 
            )

        if row.typ == 'dummy':
            # dummy = DummyRegressor(strategy='median')
            y_pred = np.full(x_test.size, y_test.mean())
            # y_pred = dummy.predict(x_test[:, np.newaxis])

        rmse = np.sqrt(mse(y_test, y_pred))
        res_mae = mae(y_test, y_pred)

        param['typ'] = row.typ
        param['fit_step'] = 'test'
        param['id_fit'] = row.id_fit
        param['coefs'] = row.coefs
        param['rmse'] = rmse
        param['mae'] = res_mae

        params.append(param)

    return pd.DataFrame(params)

df_fit = fit_compare()

plt.bar(df_fit.index, df_fit.rmse, color='red')
plt.xticks(df_fit.index, df_fit.typ)
# plt.show()          

df_test = fit_test(df_fit)
plt.bar(df_test.index, df_test.rmse, color='blue')
# plt.xticks(df_test.index, df_test.typ)
plt.show() 

df_fit.mae
