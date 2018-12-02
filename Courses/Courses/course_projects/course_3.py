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

import seaborn as sns
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
# Warm-up - TASK 1
#-------------------------------------------------------------------------



#-------------------------------------------------------------------------
def evaluate_model_poly(degree, x, y, results,
                        is_ridge = False, alpha_ridge = 0.04, plot = False):
    reg = None

    poly = PolynomialFeatures(degree, include_bias=False)

    poly.fit(x)
    X = poly.fit_transform(x)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0)

    if is_ridge :
        if alpha_ridge is None:
            reg = Ridge()
        else:
            reg = Ridge(alpha_ridge)
    else :
        reg = LinearRegression()

    reg.fit(X_tr, y_tr)
        
    y_tr_pred = reg.predict(X_tr)
    y_te_pred = reg.predict(X_te)

    # draw the model
    x_model = np.linspace(min(x), max(x), num=100)
    x_model = x_model[:, np.newaxis]

    X_model = poly.transform(x_model)
    y_model = reg.predict(X_model)
    
    row = {}
    row['is_ridge'] = is_ridge
    row['L2'] = np.sum(reg.coef_ ** 2)
    row['degree'] = degree
    row['alpha'] = alpha
    row['mse_tr'] = mse(y_tr_pred, y_tr)
    row['mse_te'] = mse(y_te_pred, y_te)
    row['score_tr'] = reg.score(X_tr, y_tr)
    row['score_te'] = reg.score(X_te, y_te)

    if plot:
        plt.plot(x_model, y_model)
        plt.scatter(X_tr[:, 0], y_tr, label='train set')
        plt.scatter(X_te[:, 0], y_te, label='test set')
              
    results.append(row)

#-------------------------------------------------------------------------

data_file = data_dir.joinpath('bike-sharing-simple.csv')
data_df = pd.read_csv(data_file)

x_ = data_df['temp'].values
y = data_df['users'].values

x_model = np.linspace(min(x_), max(x_), num=100)
coefs = np.polyfit(x_, y, 10)
y_model = np.polyval(coefs, x_model)
plt.plot(x_model, y_model)
y_pred = np.polyval(coefs, x)
plt.scatter (x_, y)

np.sqrt(mse(y, y_pred))

#-------------------------------------------------------------------------

x = x_[:, np.newaxis]

results = []
for degree in np.arange(10, 11):
    for alpha in np.logspace(-10, 0, num=100):
        evaluate_model_poly(degree, x, y, results, True, alpha, False)
        # evaluate_model_poly(degree, x, y, results, False, None, False)

plt.legend()
plt.show()

df_results = pd.DataFrame(results)
df_results

plt.semilogx(df_results.alpha, df_results.mse_tr, label='train curve')
plt.semilogx(df_results.alpha, df_results.mse_te, label='test curve')
plt.legend()
plt.show()

# determine the otpimum

df_results
df_results.iloc[df_results.mse_te.idxmin(),]

evaluate_model_poly(10, x, y, results, True, 0.00475081, True)
plt.legend()
plt.show()

#-------------------------------------------------------------------------
    
