import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid

from sklearn.model_selection import train_test_split

working_dir = os.getcwd()
working_dir = os.path.join(working_dir,'course_projects', 'module_4')
  
#-------------------------------------------------------------------------------

#age - the age of the patient
#trestbps - the resting blood pressure in mm Hg
#chol - the amount of cholesterol in mg/dl
#thalach - maximum heart rate during the tests
#oldpeak - another measure obtained using an electrocardiogram
#ca - the number of major vessels colored by fluoroscopy
# Documentation
# https://archive.ics.uci.edu/ml/datasets/heart+Disease

data_file = os.path.join(working_dir, 'heart-disease.csv')

df_data = pd.read_csv(data_file)

target = 'disease'
y = df_data[target]

# sub_columns = ['age', 'sex']

df_data_e = df_data[sub_columns]
df_data_e = df_data.drop(columns=[target])
df_data_e = pd.get_dummies(df_data_e)

X = df_data_e.values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0)

y.describe()
y_te.describe()
y_tr.describe()

#-------------------------------------------------------------------------------
# baseline most frquent =  absent; in other words, 
# with this baseline, all items will be classified as absent 
# 
# absent frequency = 49 in test (size 91) set, hence accuracy = 49 / 91 

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_tr, y_tr)

y_pre = baseline.predict(X_te)
pd.Series(y_pre).value_counts()
accuracy = baseline.score(X_te, y_te)

#-------------------------------------------------------------------------------
# KNN
#-------------------------------------------------------------------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

#-------------------------------------------------------------------------------
# tune the model 

k_values = np.arange(1, 21)

# distance : the closest, the more weight
# uniform : all the same weight
weights_functions = ['uniform', 'distance']
distance_types = [1, 2]

grid = ParameterGrid({
    'knn__n_neighbors': k_values,
    'knn__weights': weights_functions,
    'knn__p': distance_types
})

test_scores = []
for params_dict in grid:
    # Set parameters
    pipe.set_params(**params_dict)

    # Fit a k-NN classifier
    pipe.fit(X_tr, y_tr)

    # Save accuracy on test set
    params_dict['accuracy'] = pipe.score(X_te, y_te)
 
    # Save result
    test_scores.append(params_dict)

df_scores = pd.DataFrame(test_scores).sort_values(by ='accuracy', ascending=False)
best_params = df_scores.iloc[0,:][1:]

pipe.set_params(**best_params)
pipe.fit(X_tr, y_tr)

for i in np.arange(1,70):
    a_patient = {'age': 54, 'sex':'male'}
    df_patient = pd.DataFrame(a_patient, index=[1])
    df_patient = pd.get_dummies(df_patient)

    df_patient = df_patient.reindex(columns=df_data_e.columns)
    df_patient.fillna(0, inplace = True)
    df_patient = df_patient.astype(int)

    # format a_patient so that it 
    pipe.predict(df_patient.values)

# TODO : draw decisions boundaries 

#-------------------------------------------------------------------------------
# LOGIT 
#-------------------------------------------------------------------------------

# Create cross-validation object
grid_cv = GridSearchCV(LogisticRegression(multi_class='ovr', solver='liblinear'), {
    'C': [0.1, 1, 10]
}, cv=10)

grid_cv.fit(X_tr, y_tr)
df_scores = pd.DataFrame.from_dict({
    'mean_te' : grid_cv.cv_results_['mean_test_score'],
    'std_te' : grid_cv.cv_results_['std_test_score']})

df_scores.sort_values(by='mean_test_score', ascending=False)

grid_cv.predict(df_patient.values)
grid_cv.predict_proba(df_patient.values)


#-------------------------------------------------------------------------------
# SOFTMAX 
#-------------------------------------------------------------------------------

pipe = Pipeline([
    ('scaler', None), # Optional step
    ('logreg', LogisticRegression())
])

# Create cross-validation object
grid_cv = GridSearchCV(pipe, [{
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

grid_cv.fit(X_tr, y_tr)
df_scores = pd.DataFrame.from_dict({
    'mean_te' : grid_cv.cv_results_['mean_test_score'],
    'std_te' : grid_cv.cv_results_['std_test_score']})

df_scores.sort_values(by='mean_te', ascending=False)

grid_cv.predict(df_patient.values)
grid_cv.predict_proba(df_patient.values)
