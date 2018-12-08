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
import itertools
from scipy.linalg import lstsq
from scipy import stats

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
# perfoms lstsq on all features combinaisons
#-------------------------------------------------------------------------

results = []
def compute_combination(data_df, target, features):

    res = {'features' : [], 'w' : 0, 'rss' : 0, 'cn' : 0}

    target_values = data_df[target].values
    
    data_df.drop([target], axis=1)
    data_df = data_df[features]
  
    X = data_df.values
    X1 = np.c_[np.ones(X.shape[0]), X]

    w, rss, _, _ = lstsq(X1, target_values)
    cn = np.linalg.cond(X1)

    res['features'] = features
    res['w'] = w
    res['rss'] = rss
    res['cn'] = cn

    return res

#combs = []
#for i in np.arange(1, len(features)+1):
#    els = [list(x) for x in itertools.combinations(features, i)]

#    for el in els:
#        res = compute_combination(data_df, target, el)
#        results.append(res)

# removes all rss = [] which signifies that rank is defficient

# df_results = pd.DataFrame(results)
# df_results = df_results[df_results.rss > 0]

#-------------------------------------------------------------------------
# Warm-UP - TASK 1
#-------------------------------------------------------------------------

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
# POLYFIT vs GRADIENT DESCENT

# Polyfit : 
#    analytical solution (implement OLS resolution)
#    finds a polynomial response (y) to problem with one feature (x)

# Gradient descent : 
#    general iterative algorithm 
#    apply to multi-features problems


#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Warm-UP - TASK 2
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# give an idea of what we plot
#-------------------------------------------------------------------------

def plot_df(data_df) :
    plt.scatter(data_df.x1, data_df[target], label = 'x1', color = 'red')
    plt.scatter(data_df.x2, data_df[target], label = 'x2',color = 'blue')
    plt.scatter(data_df.x3, data_df[target], label = 'x3',color = 'green')
    plt.legend()
    plt.show()

#-------------------------------------------------------------------------
def evaluate_model(reg_type, data_df, features, target, results):

    res = {}

    X = data_df[features].values
    y = data_df[target].values
    y = np.c_[y]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=0.2, test_size=0.8, random_state=0)

    # linear regression without removing outliers

    if reg_type == 'linear':
        reg = LinearRegression()
    else:
        if reg_type == 'huber':
            reg = HuberRegressor(1.35)
            y_tr = y_tr.flatten()
    
    reg.fit(X_tr, y_tr)

    y_tr_pred = reg.predict(X_tr)
    y_te_pred = reg.predict(X_te)

    # metrics
    
    res['reg_type'] = reg_type
    res['mae_tr'] = mae(y_tr, y_tr_pred)
    res['mae_te'] = mae(y_te, y_te_pred)

    results.append(res)

    # finds out that (without shuffle) test scores better than train mse !
    # because of the outliers, I guess

#-------------------------------------------------------------------------

data_file = data_dir.joinpath('task-2.csv')
data_df = pd.read_csv(data_file)

target = 'y'
features = data_df.columns.tolist()
features.remove(target)

results = []
evaluate_model('linear', data_df, features, target, results)
plot_df(data_df)

# removes outliers

data_df_wo = data_df.copy()
for f in data_df.columns:
    z = (data_df_wo[f] - data_df_wo[f].mean()) / data_df_wo[f].std()
    outliers = np.abs(z) >= 2

    print (f, outliers.sum())
    data_df_wo = data_df_wo[~outliers]

plot_df(data_df_wo)

evaluate_model('linear', data_df_wo, features, target, results)
evaluate_model('huber', data_df, features, target, results)

df_results = pd.DataFrame(results)
df_results

#-------------------------------------------------------------------------
# Warm-UP - TASK 3
#-------------------------------------------------------------------------
def evaluate_model_poly(degree, x, y, is_ridge, alpha_ridge, results, plot = False):
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
    row['L2']       = np.sum(reg.coef_ ** 2)
    row['degree']   = degree
    row['alpha']    = alpha_ridge
    row['mse_tr']   = mse(y_tr_pred, y_tr)
    row['mse_te']   = mse(y_te_pred, y_te)
    row['score_tr'] = reg.score(X_tr, y_tr)
    row['score_te'] = reg.score(X_te, y_te)

    if plot:
        plt.plot(x_model, y_model, label='model : ' + str("ridge regression = ") + str(is_ridge) + "; degree = " + str(degree) )
        plt.scatter(X_tr[:, 0], y_tr, label='train set')
        plt.scatter(X_te[:, 0], y_te, label='test set')
              
    results.append(row)

#-------------------------------------------------------------------------

data_file = data_dir.joinpath('task-3.csv')
data_df = pd.read_csv(data_file)

x_ = data_df['x'].values
y = data_df['y'].values

#-------------------------------------------------------------------------
# Regression using polynomial features 
#-------------------------------------------------------------------------
x = x_[:, np.newaxis]

results = []
evaluate_model_poly(10, x, y, False, None, results, True)
plt.legend()
plt.show()

df_results = pd.DataFrame(results)
df_results

#-------------------------------------------------------------------------
# Regression using Ridge(alpha) / grid search
#-------------------------------------------------------------------------
results = []
for degree in np.arange(10, 11):
    for alpha in np.logspace(-2, 10, num=100):
        evaluate_model_poly(degree, x, y, True, alpha, results, False)
      
df_results = pd.DataFrame(results)
df_results

#-------------------------------------------------------------------------
# Plots mse train / test
#-------------------------------------------------------------------------
plt.semilogx(df_results.alpha, df_results.mse_tr, label='mse train curve')
plt.semilogx(df_results.alpha, df_results.mse_te, label='mse test curve')
plt.legend()
plt.show()

#-------------------------------------------------------------------------
# determine the otpimum
#-------------------------------------------------------------------------
df_results
df_results.iloc[df_results.mse_te.idxmin(),]

#-------------------------------------------------------------------------
# draw the optimum ridge solution and compares with 
# a polynomial regression of degree 4 
#-------------------------------------------------------------------------
results = []
evaluate_model_poly(10, x, y, True, 0.162975, results, True)

# If we compare with a polynomial regression of degree 4 
evaluate_model_poly(4, x, y, False, None, results, True)

plt.legend()
plt.show()

df_results = pd.DataFrame(results)
df_results

#-------------------------------------------------------------------------




