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
from sklearn.linear_model import LinearRegression
import itertools
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------
# répertoire de travail
#-------------------------------------------------------------------------

working_dir = PureWindowsPath(os.getcwd())
data_dir = PureWindowsPath(working_dir.joinpath('Data').joinpath('module_3'))
data_result_name = 'result.csv'

pd.set_option('display.max_columns', 30)

def mae(y, y_pred):
    return np.mean(np.abs(y-y_pred))

def mse(y, y_pred):
    return np.mean(np.square(y - y_pred))

#-------------------------------------------------------------------------

def plot_models(data_df):

    i_c = 0
    for i in np.arange(len(data_df.temp)):

        i_c += 1
        color=sns.color_palette()[np.mod(i_c, 6)]
        plt.scatter(data_df.temp[i].values, data_df.y_te[i].values, color=color, label=data_df.tag[i])

        if data_df.prediction[i]:

            i_c += 1
            color=sns.color_palette()[np.mod(i_c, 6)]
            plt.scatter(data_df.temp[i].values, data_df.y_pred_te[i].values, color=color, label=data_df.tag[i] + 
                         str(" prediction"))
   
    plt.legend()
    plt.show()

#-------------------------------------------------------------------------

def evaluate_model(data_df, target, tag, prediction):
    
    col_list = data_df.columns.copy()
    col_list_wo_target = col_list.drop(target)

    X = data_df[col_list_wo_target].values
    y = data_df[target].values
    y = np.c_[y]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size = 0.5, test_size = 0.5, random_state=1)

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)

    # negatives values not allowed 
    y_pred_tr = lr.predict(X_tr)
    y_pred_te = lr.predict(X_te)
   
    # determine the median data 

    dummy = DummyRegressor(strategy='median')
    dummy.fit(X_tr, y_tr)
    y_pred_base = dummy.predict(X_te)

    row = {}
    row['features']     = data_df.columns
    row['mae_tr']       = mae(y_pred_tr, y_tr)
    row['mae_te']       = mae(y_pred_te, y_te)
    row['mae_baseline'] = mae(y_pred_base, y_te)
    row['temp']         = pd.Series(X_te[:,0])
    row['y_te']         = pd.Series(y_te[:,0])
    row['y_pred_te']    = pd.Series(y_pred_te[:,0])
    row['prediction']   = prediction
    row['tag']          = tag

    results.append(row)

#-------------------------------------------------------------------------

data_file = data_dir.joinpath('bike-sharing-simple.csv')
data_df = pd.read_csv(data_file)

x = data_df.temp.values
y = data_df.casual.values

results = []
evaluate_model(data_df, 'y', "", True)

evaluate_model(data_df_1, 'y', "", True)

df_results = pd.DataFrame(results)
plot_models(df_results)
df_results[['tag', 'mae_tr', 'mae_te', 'mae_baseline']]

#-------------------------------------------------------------------------

X = np.c_[np.ones(len(x)), x]

x = data_df.x
y = data_df.y
plt.scatter(x, y, marker='o')

coefs = np.polyfit(x, y, deg=10)

# Compute prediction curve
x_values = np.linspace(min(x), max(x), num=50)
y_values = np.polyval(coefs, x_values)

plt.scatter(x_values, y_values,marker='^')
plt.show()

#-------------------------------------------------------------------------

from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Create the polynomial features
poly_obj = PolynomialFeatures(degree=10, include_bias=False)

f_ = poly_obj.fit(x[:, np.newaxis])

X_poly = poly_obj.fit_transform(x[:, np.newaxis])

X_tr, X_te, y_tr, y_te = train_test_split(
    X_poly, y, test_size=25, random_state=0)

print('Train set:', X_tr.shape, y_tr.shape) # (25, 10) (25,)
print('Test set:', X_te.shape, y_te.shape) # (25, 10) (25,)

lr.fit(X_tr, y_tr)

# Plot the model
x_values = np.linspace(min(x), max(x), num=100)
x_values_poly = poly_obj.transform(x_values[:, np.newaxis])
y_values_lr = lr.predict(x_values_poly)

plt.scatter(X_tr[:, 0], y_tr, label='train set')
plt.scatter(X_te[:, 0], y_te, label='test set')
plt.plot(x_values, y_values_lr)
plt.legend()
plt.show()

from sklearn.linear_model import Ridge

# Ridge regression
ridge = Ridge()
ridge.fit(X_tr, y_tr)

# Plot the model
y_values_ridge = ridge.predict(x_values_poly)

plt.scatter(X_tr[:, 0], y_tr, label='train set')
plt.scatter(X_te[:, 0], y_te, label='test set')
plt.plot(x_values, y_values_ridge)
plt.legend()
plt.show()

ridge.coef_
