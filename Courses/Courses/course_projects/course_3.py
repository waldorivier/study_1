#-------------------------------------------------------------------------
# Module 3 : course project
#-------------------------------------------------------------------------

import os
from pathlib import PureWindowsPath
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

from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------
# working directory
#-------------------------------------------------------------------------

working_dir = PureWindowsPath(os.getcwd())
data_dir = PureWindowsPath(working_dir.joinpath('Data').joinpath('module_3'))
data_result_name = 'result.csv'

pd.set_option('display.max_columns', 30)

#-------------------------------------------------------------------------
# Warm-UP - TASK 1
#-------------------------------------------------------------------------
def gradient(x, y, results, lr = None):

    if lr is None:
        lr = 0.8

    a, b = 0, 0
    n_steps = 400

    # Gradient descent
    log_a = [a]
    log_b = [b]

    for step in range(n_steps):
        # Compute partial derivatives
        y_pred = a * np.log(x) + b
        error = y - y_pred
        a_grad = -2*np.mean(x*error)
        b_grad = -2*np.mean(error)

        # Update parameters
        a -= lr*a_grad
        b -= lr*b_grad

        # Log a, b values

        row = {}

        row['a'] = a
        row['b'] = b

        results.append(row)

#-------------------------------------------------------------------------
data_file = data_dir.joinpath('task-1.csv')
data_df = pd.read_csv(data_file)

x = data_df.iloc[:,0]
y = data_df.iloc[:,1]

#-------------------------------------------------------------------------
# variable change x -> _x = log(x)
_x = np.log(x)

#-------------------------------------------------------------------------
# fits a linear regression  
coefs = np.polyfit(_x, y, deg = 1)
y_pred = np.polyval(coefs, _x)

#-------------------------------------------------------------------------
# prints the model 
x_model = np.linspace(min(_x), max(_x), num=100)
y_model = np.polyval(coefs, x_model)

plt.scatter(x, y, label='observations', color='blue')
plt.plot(np.exp(x_model), y_model, label = 'model', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#-------------------------------------------------------------------------
# metrics 
mse(y, y_pred)

# compute baseline (default strategy = mean)
_X = np.c_[_x]
d_reg = DummyRegressor()
d_reg.fit(_X, y)
y_pred_bsl = d_reg.predict(_x)
mse(y, y_pred_bsl)

#-------------------------------------------------------------------------
# Compares metrics with R2 scores 
# total error - model error / total error => 1 - model error / total error

R2 = 1 - mse(y, y_pred) / mse(y, y_pred_bsl)

#-------------------------------------------------------------------------

results = []
gradient(x, y, results, 0.1)
df_results = pd.DataFrame(results)

