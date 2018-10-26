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

def MSE(y, y_pred):
    assert y.size == y_pred.size

    return np.mean(np.square(y - y_pred))

    return mse
    
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

assert_almost_equal(MSE(y_test, y_test_pred1), 0, decimal=5)
assert_almost_equal(MSE(y_test, y_test_pred2), 3 , decimal=5)
assert_almost_equal(MSE(y_test, y_test_pred3), 6, decimal=5)

print('tests passed!')

#------------------

ar_rmse = []

x_values = np.linspace(x.min(), x.max(), num=x.size)
plt.scatter(x, y, label=None)
for i in np.arange(1, 10, 1):
    
    coefs = np.polyfit(x, y, deg=i)
    y_values = np.polyval(coefs, x_values)
   
    plt.plot(x_values, y_values, label="polyfit de degré (" + str(i) + ")" )
  
    ar_rmse.append(np.sqrt(MSE(y, y_values)))

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

