import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


from sklearn.tree import export_graphviz
import graphviz
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

import itertools
import math
import random as r  

#-------------------------------------------------------------------------------

sns.set()

# Get a few colors from the default color palette
blue, green, red = sns.color_palette()[:3]

# Helper function
def decision_surface(x1, x2, y, estimator):
    # Create figure
    fig = plt.figure(figsize=(6, 6))
    axes = fig.gca() # Get the current axes

    # Same scale for x- and y-axis
    axes.set_aspect('equal', adjustable='box')

    # Plot data points
    class1_idx = (y == 1)

    plt.scatter(x1[class1_idx], x2[class1_idx],
        color=red, label='class 1', s=12)
    plt.scatter(x1[~class1_idx], x2[~class1_idx],
        color=blue, label='class 0', s=12)

    # Create a grid of values
    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    x_values = np.linspace(*xlim, num=500)
    y_values = np.linspace(*ylim, num=500)
    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = np.c_[xx.flatten(), yy.flatten()]

    # Compute predictions
    preds = estimator.predict(grid_points)
    zz = preds.reshape(xx.shape)

    # Draw decision boundary
    plt.contour(xx, yy, zz, levels=[0.5], colors='gray')

    # Plot decision surface
    plt.contourf(xx, yy, zz, alpha=0.1, cmap=plt.cm.coolwarm)

    # Show labels on a white frame
    plt.legend(frameon=True, facecolor='white')
    plt.show()

#-------------------------------------------------------------------------------

dir = os.getcwd()
dir = os.path.join(dir,'course_projects', 'module_4')

data_df = pd.read_csv(os.path.join(dir, 'spirals.csv'))

y = data_df.iloc[0:,-1].values
X = data_df.iloc[0:,0:-1].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0)

#-------------------------------------------------------------------------------
# A. LOGIT
#-------------------------------------------------------------------------------

pipe = Pipeline([
    ('scaler', None), # Optional step
    ('logreg', LogisticRegression())
])

# Create cross-validation object
estimator = GridSearchCV(pipe, [{
    'logreg__multi_class': ['ovr'],
    'logreg__C': [0.1, 1, 10],
    'logreg__solver': ['liblinear']
}, {
    'scaler': [StandardScaler()],
    'logreg__multi_class': ['multinomial'],
    'logreg__C': [0.1, 1, 10],
    'logreg__solver': ['saga'],
    'logreg__max_iter': [1000],
    'logreg__random_state': [0]
}], cv=10)

estimator.fit(X_tr, y_tr)
estimator.score(X_te, y_te)

#-------------------------------------------------------------------------------

C = np.logspace(-4, 4, num=10)

estimator = LogisticRegressionCV(C, cv=10, multi_class='multinomial', solver='saga')
estimator.fit(X_tr, y_tr)
estimator.score(X_te, y_te)

decision_surface(X_te[0:,0], X_te[0:,1], y_te, estimator)

#-------------------------------------------------------------------------------
# REM : both grid deliver the same results; not surprising because the second ones is
#       a digest of the former
#       In this case, we have only a binary output, hence a softmax is not mandatory 
#       
#       C : to pilot the regularisazion term : 
#     
#       L2 (penalization term) ia added to the LogisticRegression
#       => regularization = C * loss + (alpha) * L2   
#                
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# B. SVM
#-------------------------------------------------------------------------------

# Create SVM with linear kernel
C = np.logspace(-4, 0, num=10)

pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('svm', LinearSVC())
])

estimator = GridSearchCV(pipe, [{
    'svm__C': C
}], cv=10)


# Fit estimator
estimator.fit(X_tr, y_tr)
estimator.score(X_te,y_te)

decision_surface(X_tr[0:,0], X_tr[0:,1], y_tr, estimator)
decision_surface(X_te[0:,0], X_te[0:,1], y_te, estimator)

#-------------------------------------------------------------------------------
# C. KNN
#-------------------------------------------------------------------------------

C = np.arange(5, 6)

pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('knn', KNeighborsClassifier())
])

estimator = GridSearchCV(pipe, [{
    'knn__n_neighbors': C
}], cv=10)

estimator.fit(X_tr, y_tr)
estimator.score(X_te,y_te)

decision_surface(X_tr[0:,0], X_tr[0:,1], y_tr, estimator)
decision_surface(X_te[0:,0], X_te[0:,1], y_te, estimator)

#-------------------------------------------------------------------------------
# D. Decision TREE
#-------------------------------------------------------------------------------

pipe = Pipeline([
    ('scaler', None), 
    ('tree', DecisionTreeClassifier())
])

estimator = GridSearchCV(pipe, [{
}], cv=10)

estimator.fit(X_tr, y_tr)
estimator.score(X_te,y_te)

decision_surface(X_tr[0:,0], X_tr[0:,1], y_tr, estimator)
decision_surface(X_te[0:,0], X_te[0:,1], y_te, estimator)

p = estimator.get_params()['estimator']
e = p.steps[1][1]
e.fit(X_tr, y_tr)

dot_data = export_graphviz(
    e, out_file='tree.dot',
    feature_names=data_df.columns[:-1], class_names=['1', '0'],
    filled=True, rounded=True, proportion=True   
)

s = graphviz.Source(dot_data)

from graphviz import Source
path = 'tree.dot'
s = Source.from_file(path)
# s.view()

#-------------------------------------------------------------------------------
# E. Random Forest
#-------------------------------------------------------------------------------

pipe = Pipeline([
    ('scaler', None), 
    ('tree', RandomForestClassifier())
])

n_trees = np.arange(2, 20, 2)

estimator = GridSearchCV(pipe, [
    {'scaler' : [StandardScaler()], 'tree__n_estimators' : n_trees }, 
    {'tree__n_estimators' : n_trees }
    ], cv=10)

estimator.fit(X_tr, y_tr)
estimator.score(X_te,y_te)

decision_surface(X_tr[0:,0], X_tr[0:,1], y_tr, estimator)
decision_surface(X_te[0:,0], X_te[0:,1], y_te, estimator)

#-------------------------------------------------------------------------------
# F. SVM RBF 
#-------------------------------------------------------------------------------

pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('svm', SVC())
])

n_gamma = np.logspace(-2, 0, 10)
C = np.logspace(-4, 0, num=10)

estimator = GridSearchCV(pipe, [{'svm__C' : C, 'svm__gamma':n_gamma}
                                ], cv=10)

estimator.fit(X_tr, y_tr)
estimator.score(X_te,y_te)

decision_surface(X_tr[0:,0], X_tr[0:,1], y_tr, estimator)
decision_surface(X_te[0:,0], X_te[0:,1], y_te, estimator)