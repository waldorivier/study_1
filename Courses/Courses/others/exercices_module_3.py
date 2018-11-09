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
# 3.2.2
#-------------------------------------------------------------------------

from sklearn.datasets import load_boston
data = load_boston()

# Fit a linear regression model
import numpy as np
x = data['data'][:, -1] # Extract LSTAT feature
y = data['target']
coefs = np.polyfit(np.log(x), y, deg=1)

# Compute prediction curve
x_values = np.linspace(min(x), max(x), num=100)
y_values = np.polyval(coefs, np.log(x_values))

# Plot predictions

import matplotlib.pyplot as plt
plt.scatter(x, y, label=None)
plt.plot(x_values, y_values, label='linear regression', color='xkcd:strawberry')
plt.title('Predicting median value of owner-occupied homes')
plt.xlabel('% lower status of the population')
plt.ylabel('Median value in $1000\'s')
plt.legend()
plt.show()

#-------------------------------------------------------------------------
# 3.2.3
#-------------------------------------------------------------------------

data_file =  data_dir.joinpath('data-points-1.csv')
data_df = pd.read_csv(data_file)

x = data_df.x.values
y = data_df.y.values

sns.set()

plt.scatter(x, y)
plt.show()

coefs = np.polyfit(x, y, deg=1)
print('Coefficients:', coefs)
a, b = coefs

# divide badwidth with in 100 even space intervals
x_values = np.linspace(0, 1, num=100)

y_values = a*x_values + b
plt.scatter(x, y, label=None)
plt.plot(x_values, y_values, color='red', label='polyfit(deg=1)')
plt.legend()
plt.show()

#------------------

data_file =  data_dir.joinpath('data-points-2.csv')
data_df = pd.read_csv(data_file)

x2 = data_df.x.values
y2 = data_df.y.values

plt.scatter(x2, y2, label=None)
plt.show()

x_values2 = np.linspace(0, 1, num=100)
coefs2 = np.polyfit(x2, y2, deg=3)
y_values2 = np.polyval(coefs2, x_values2)

plt.scatter(x2, y2, label=None)
plt.plot(x_values2, y_values2, color='red', label='polyfit(deg=3)')
plt.legend()
plt.show()

#------------------

data_file =  data_dir.joinpath('data-points-3.csv')
data_df = pd.read_csv(data_file)

x3 = data_df.x.values
y3 = data_df.y.values

x_values3 = np.linspace(0, x3.max(), num=100)
coefs3 = np.polyfit(np.log(x3), y3, deg=1)
y_values3 = np.polyval(coefs3, np.log(x_values3))

plt.scatter(x3, y3, label=None)
plt.plot(x_values3, y_values3, color='red', label='polyfit(log)')
plt.legend()
plt.show()  

#-------------------------------------------------------------------------
# 3.2.5
#-------------------------------------------------------------------------

def RSS(y, y_pred):
    assert y.size == y_pred.size

    return np.sum(np.square(y - y_pred))

data_file =  data_dir.joinpath('marketing-campaign.csv')
data_df = pd.read_csv(data_file)

target = 'sales'
features = data_df.columns.tolist()
features.remove(target)

sns.pairplot(data_df, 
             x_vars=features,
             y_vars=[target], 
             kind="reg",
             plot_kws={'line_kws':{'color':'red'}})
plt.show()

X = data_df.drop('sales', axis=1).values

x = data_df.tv.values
y = data_df.sales.values

coefs = np.polyfit(x, y, deg=1)
x_values = np.linspace(x.min(), x.max(), num=50)
y_values = np.polyval(coefs, x_values)

plt.scatter(x, y, label=None)
plt.plot(x_values, y_values)
plt.show() 

RSS(y, y_values)

#-------------------------------------------------------------------------
# 3.2.8
#-------------------------------------------------------------------------

def mse(y, y_pred):
    return np.mean(np.square(y - y_pred))
    
data_file =  data_dir.joinpath('bike-sharing-simple.csv')
data_df = pd.read_csv(data_file)

x = data_df.temp.values
y = data_df.users.values

target = 'users'
features = data_df.columns.tolist()
features.remove(target)

sns.pairplot(data_df, 
             x_vars=features,
             y_vars=[target], 
             kind="reg",
             plot_kws={'line_kws':{'color':'red'}})
plt.show()

#------------------
# Evaluate the MSE function

from numpy.testing import assert_almost_equal

y_test = np.array([1, 2, 3])
y_test_pred1 = np.array([1, 2, 3])
y_test_pred2 = np.array([1, 5, 3])
y_test_pred3 = np.array([1, 5, 6])

assert_almost_equal(mse(y_test, y_test_pred1), 0, decimal=5)
assert_almost_equal(mse(y_test, y_test_pred2), 3 , decimal=5)
assert_almost_equal(mse(y_test, y_test_pred3), 6, decimal=5)

print('tests passed!')

#------------------

ar_rmse = []

x_values = np.linspace(x.min(), x.max(), num=x.size)
plt.scatter(x, y, label=None)
for i in np.arange(1, 10, 1):
    
    coefs = np.polyfit(x, y, deg=i)
    y_values = np.polyval(coefs, x_values)
   
    plt.plot(x_values, y_values, label="polyfit de degré (" + str(i) + ")" )
  
    ar_rmse.append(np.sqrt(mse(y, y_values)))

plt.legend()
plt.show() 

ar_rmse
plt.scatter(np.arange(len(ar_rmse)), ar_rmse, label=None)
plt.show() 

# obtenir la valeur min
ar_rmse.sort()
ar_rmse[0]

#-------------------------------------------------------------------------
# 3.3.2 OUTLIERS
#-------------------------------------------------------------------------

data_file =  data_dir.joinpath('marketing-campaign-with-outliers.csv')
data_df = pd.read_csv(data_file)

target = 'sales'
features = data_df.columns.tolist()
features.remove(target)

sns.pairplot(data_df, 
             x_vars=features,
             y_vars=[target], 
             kind="reg",
             plot_kws={'line_kws':{'color':'red'}})
plt.show()

x = data_df.tv.values
y = data_df.sales.values

plt.scatter(x[-5:], y[-5:]) # outliers
plt.scatter(x[:-5], y[:-5]) # other points
plt.show()

# Fit a linear regression
coefs = np.polyfit(x, y, deg=1)
print('coefs:', coefs) # [ 0.20613307  2.76540858]

# Fit a linear regression without the 5 outliers
coefs2 = np.polyfit(x[:-5], y[:-5], deg=1)
print('coefs2:', coefs2) # [ 0.42063597  1.27867727]

# filter outliers in order to extract them
# définit un tableau de boolean
idx = (((x < 4) & (y > 6)) | ((x > 10) & (y < 2)))

# wo outliers
x1, y1 = x[~idx], y[~idx]
plt.scatter(x1, y1) # outliers
plt.show()

# only outliers
x1, y1 = x[idx], y[idx]
plt.scatter(x1, y1) # outliers
plt.show()

# metric to remove outliers
z_scores = (y - y.mean()) / y.std()
plt.scatter(x, y, c = z_scores, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.show()

pd.Series(z_scores).hist()
plt.show()

idx = (np.abs(z_scores) > 2)
idx

# wo outliers
x1, y1 = x[~idx], y[~idx]
plt.scatter(x1, y1) # outliers
plt.show()

#-------------------------------------------------------------------------
# 3.3.3 SURFACES
#-------------------------------------------------------------------------

def mae(y, y_pred):
    return np.mean(np.abs(y-y_pred))

points = np.array([[1, 2], [0, 1], [-1.5, 0]])
x, y = points[:, 0], points[:, 1]

plt.scatter(x, y)
plt.show()

#-------------------------------------------------------------------------
# 3.3.4 HUBERT LOSS
#-------------------------------------------------------------------------

data_file =  data_dir.joinpath('marketing-campaign-with-outliers.csv')
data_df = pd.read_csv(data_file)

x = data_df.tv.values
y = data_df.sales.values

lr_huber = SGDRegressor(loss='huber', penalty='none', epsilon=1, max_iter=10000)

# transfom vector to matrix of (x, 1)
x = x[:, np.newaxis]
lr_huber.fit(x, y)

lr_huber.coef_
lr_huber.intercept_

x_values = np.linspace(min(x), max(x), num=100) # Shape (100,)
y_values_huber = lr_huber.predict(
    x_values[:, np.newaxis] # Shape (100,1)
)

blue, green, red = sns.color_palette()[:3]

plt.scatter(x, y, color=blue)
plt.plot(x_values, y_values_huber, color=red)
# plt.show()

lr_squared = SGDRegressor(loss='squared_loss', penalty='none', max_iter=10000)

lr_huber = HuberRegressor(epsilon=1.35)
lr_huber.fit(x, y)
y_values_huber = lr_huber.predict(
    x_values[:, np.newaxis] # Shape (100,1)
)
plt.scatter(x, y, color=blue)
plt.plot(x_values, y_values_huber, color=red)
# plt.show()

#-------------------------------------------------------------------------
# 3.3.5 EXERCICES
#-------------------------------------------------------------------------

data_file = data_dir.joinpath('brain-and-body-weights.csv')
data_df = pd.read_csv(data_file)

# linear regression

data_df_ = data_df.copy()
data_df_.body = np.log(data_df_.body)
data_df_.brain = np.log(data_df_.brain)

# helper function to plot with labels'point with seaborn
def plot_(data_df : pd.DataFrame):

    pl = sns.regplot(data=data_df, x='body', y='brain') 

    # prints labels in front of each points
    for i in range(0, data_df.shape[0]):
        pl.text(data_df.body[i] + 0.2, data_df.brain[i], data_df.label[i], 
                horizontalalignment='left', size='small', color='black')

# plot all datas 
plot_(data_df_)

# define outliers
# a, b = np.polyfit(data_df_.body.values, data_df_.brain.values, deg=1)
# i_outliers = data_df_.brain < (data_df_.body * a - 3.6)
i_outlier = data_df_.body > 16

# remove outliers
df_wo_outliers = data_df_[~i_outliers]

# reindex the df
df_wo_outliers.index = pd.RangeIndex(len(df_wo_outliers.index))

# plot wo outliers
plot_(df_wo_outliers)
# plt.show()

#------------------------------------------------------------------------------
# linear regression with Huber loss
#------------------------------------------------------------------------------

data_file =  data_dir.joinpath('brain-and-body-weights.csv')
data_df = pd.read_csv(data_file)

# linear regression

data_df_ = data_df.copy()
data_df_.body = np.log(data_df_.body)
data_df_.brain = np.log(data_df_.brain)

x = data_df_.body.values
y = data_df_.brain.values

x_values = np.linspace(data_df_.body.min(), data_df_.body.max(), 100)

ar_lr_huber_coef = []

plt.scatter(x, y)
i = 0

# ep increase => the regression line get closer the outliers

try:
    for ep in np.linspace(1.1, 1.5, 8):
    
        lr_huber = HuberRegressor(epsilon=ep)
        lr_huber.fit(x[:, np.newaxis], y)
        y_values_huber = lr_huber.predict(
            x_values[:, np.newaxis] 
        )

        t_hr = (ep, lr_huber.coef_, lr_huber.intercept_)
        ar_lr_huber_coef.append(t_hr)
    
        plt.plot(x_values, y_values_huber, 
                 color=sns.color_palette()[np.mod(i,6)], 
                 label=t_hr)
        i = i + 1

    plt.legend()        
    # plt.show()
 
except ValueError :
    print (i)

ar_lr_huber_coef
pd.DataFrame(ar_lr_huber_coef, columns=[''])

#------------------------------------------------------------------------------

            
#-------------------------------------------------------------------------
# 3.3.8 
#-------------------------------------------------------------------------

data_file =  data_dir.joinpath('bike-sharing-three-models.csv')
data_df = pd.read_csv(data_file)

# Extract variables
x = data_df.temp.values
y = data_df.users.values

# Plot the models
plt.scatter(x, y)
plt.plot(x, data_df.pred_lr, label='linear regression')
plt.plot(x, data_df.pred_poly3, label='polyfit(deg=3)')
plt.plot(x, data_df.pred_huber3, label='with Huber loss')
plt.legend()
plt.show()

y=[]

a = [1, 2, 3, 5, 6, 25]
x = pd.Series(a)

for i in np.arange(0, 25):
    y.append(np.mean(np.abs(x-i)))

plt.plot(y, label='mae')
plt.legend()
plt.show()

dummy = DummyRegressor(strategy='mean')
dummy.fit(x[:, np.newaxis], y)
        
pred_baseline = dummy.predict(x[:, np.newaxis])

#-------------------------------------------------------------------------
# 3.3.8 Exercices
#-------------------------------------------------------------------------

data_file =  data_dir.joinpath('bike-sharing-test.csv')
df_test = pd.read_csv(data_file)

data_file = data_dir.joinpath('bike-sharing-train.csv')
df_train = pd.read_csv(data_file)

df_test.sort_values(by=['temp'], inplace=True)
df_train.sort_values(by=['temp'], inplace=True)

x_train = df_train.temp.values
y_train = df_train.users.values

# remove outliers on train set

def z_score (x):
    return (x - x.mean()) / x.std()

z_scores = z_score(y_train)

plt.scatter(x_train, y_train, c=z_scores, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.show()

# remove all values with z_score > 1.5

outliers = z_scores > 1.5

df_train_wo_outliers = df_train[~outliers]
df_train_wo_outliers.index = pd.RangeIndex(len(df_train_wo_outliers.index))

x_train = df_train_wo_outliers.temp.values
y_train = df_train_wo_outliers.users.values

# fill an array with identical value
a_mean = np.full(x_train.size, y_train.mean())

# trace a line at the mean
df_train_wo_outliers.users.hist()
plt.plot([y_train.mean(), y_train.mean()], 
         [0, 50], color='red')
plt.show()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fit_dummy(x, y):
    params = []
    param = {'id_fit' : 0, 'coefs' : (), 'rmse' : 0, 'mae' : 0, 'typ' : "", 'fit_step' : ""}

    plt.scatter(x, y)

    dummy = DummyRegressor(strategy='median')
    dummy.fit(x[:, np.newaxis], y)

    y_pred = dummy.predict(x[:, np.newaxis])

    rmse = np.sqrt(mse(y, y_pred))
    res_mae = mae(y, y_pred)

    param['typ'] = 'dummy'
    param['fit_step'] = 'train'
    param['id_fit'] = i
    param['coefs'] = ''
    param['rmse'] = rmse
    param['mae'] = res_mae


    params.append(param)

    plt.plot(x, y_pred, label = param)
    plt.show()

    return pd.DataFrame(params)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fit_huber(x, y):
    
    params = []
    
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

            plt.plot(x, y_pred, 
                     color=sns.color_palette()[np.mod(i,6)], 
                     label=param)
            i = i + 1

        plt.legend()        
        plt.show()
 
    except ValueError :
        print (i)

    return pd.DataFrame(params)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def fit_poly(x, y):

    params = []
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
        
        plt.plot(x, y_pred, label=param, linewidth=1)

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

    df_compare = pd.concat([x, y, z])
    df_compare.index = pd.RangeIndex(len(df_compare.index))

    return df_compare

df_fit = fit_compare()
plt.bar(df_fit.index, df.mae)
plt.xticks(df_fit.index, df.typ)
plt.show()

#------------------------------------------------------------------------------
# Apply fit selection on test data
#------------------------------------------------------------------------------
# def evaluate_fit(df_fit):

data_file =  data_dir.joinpath('bike-sharing-test.csv')
df_test = pd.read_csv(data_file)

df_test.sort_values(by=['temp'], inplace=True)
x_test = df_test.temp.values
y_test = df_test.users.values
   
params = []
for row in df_fit.iterrows():
    
    y_pred = None;

    if row['typ'] == 'poly':
        y_pred = np.polyval(row['coefs'], x_test)
    
    if row['typ'] == 'huber':
               
        ep = row['coefs'].values[0][0]
        coef = row['coefs'].values[0][1][0]
        intercept = row['coefs'].values[0][1][1]
        
        lr_huber = HuberRegressor(epsilon=ep)
        lr_huber.coef_ = coef
        lr_huber.intercept_ = intercept

        y_pred = lr_huber.predict(
                x_test[:, np.newaxis] 
        )

    if row['typ'] == 'dummy':
        dummy = DummyRegressor(strategy='median')
        y_pred = dummy.predict(x_test[:, np.newaxis])

    rmse = np.sqrt(mse(y_test, y_pred))
    res_mae = mae(y_test, y_pred)

    param['typ'] = row['typ']
    param['fit_step'] = 'test'
    param['id_fit'] = row['id_fit']
    param['coefs'] = row['coefs']
    param['rmse'] = rmse
    param['mae'] = res_mae

    params.append(params)
           