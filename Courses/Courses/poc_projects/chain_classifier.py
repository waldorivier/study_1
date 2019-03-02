# import mysql.connector
import os
import sqlite3
# import cx_Oracle    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

#soldel_db = mysql.connector.connect(
#  host="localhost",
#  user="root",
#  passwd="waldo",
#  database="mupe"
#)

# connection = cx_Oracle.connect("Data Source = LABCIT; User ID = PADEV96_DATA; Password = PADEV96_DATA")

pd.set_option('display.max_columns', 90)

#-------------------------------------------------------------------------
# persiste / charge un DataFrame vers / de la  base de données 
# REM : db_name sans extension ; le nom de la table correspond à celui de 
#       la base de données
#-------------------------------------------------------------------------
def helper_store_df_to_db(df, dir, db_name, table_name) :
    db_file = os.path.join(dir, db_name + ".db" )
    db = sqlite3.connect(db_file)

    df.to_sql(table_name, db, chunksize=2000, if_exists='append')
    db.close()

def helper_load_df_from_db(dir, db_name, table_name) :
    db_file = os.path.join(dir, db_name + ".db")
    db = sqlite3.connect(db_file)

    df = pd.read_sql_query("select * from " + table_name, con=db)
    db.close()

    return df

def helper_store_csv_to_db(dir, data_file, db_name, table_name) :
    db_file = os.path.join(dir, db_name + ".db" )
    db = sqlite3.connect(db_file)
    
    for chunk in pd.read_csv(data_file, chunksize=2000):
        chunk.to_sql(name=table_name, con=db, if_exists="append", index=False)  

    db.close()

#-------------------------------------------------------------------------
def format_date():
    def _f(d):
        # transform to date time
        # _d = datetime.strptime(d, '%Y-%m-%d')

        # transform to string as specified
        _d = d.strftime('%d.%m.%Y')
        return _d
    return _f

f_format_date = format_date()

#-------------------------------------------------------------------------

dir = os.getcwd()
dir = os.path.join(dir, 'poc_projects')

def read_and_store(data_file):
    df_chain = pd.read_excel(data_file)
    # df_chain = pd.DataFrame.from_csv(data_file, sep='\t')
  
    df_chain['PE_CHAI_DDV'] = df_chain['PE_CHAI_DDV'].apply(f_format_date)
    
    # df_chain['PE_CHAI_DDV'] = df_chain['PE_CHAI_DDV'].str.replace(r' 00:00:00', "")
    helper_store_df_to_db(df_chain, dir, 'chain', 'chain')


if 0:
    data_file = os.path.join(dir, 'chain.xlsx')
    read_and_store(data_file)

    # data_file = os.path.join(dir, 'chain_1.xlsx')
    # read_and_store(data_file)

#-------------------------------------------------------------------------
class chain_vector:
    _df_items = None

    def __init__(self, df):
        self._df_items = df.copy()[['NOM_ELEM', 'NOM_LOGI']]
        self._df_items.drop_duplicates(inplace=True)
        self._build_logi_name()

    def compute_dist(self, other):
        if other == self:
            return -1

        if other != None:
            lunion = len(pd.merge(self._df_items, other._df_items, how='outer'))
            lintersection = len(pd.merge(self._df_items, other._df_items, how='inner'))
            
            return  1 - lintersection / lunion
    
    def _build_logi_name(self):
        self._df_items['ID_LOGI'] = self._df_items['NOM_ELEM'] + "_" + self._df_items['NOM_LOGI']
    
    def _to_elems_matrix(self, cm_ref):
        assert self._df_items['ID_LOGI'] is not None
        
        _df_elems = self._df_items['ID_LOGI']
        ones = np.ones((1, _df_elems.shape[0]))
        _df_elems = pd.DataFrame(ones, columns=_df_elems)

        if cm_ref is not None:
            _df_elems = _df_elems.reindex(columns=cm_ref.columns)
            _df_elems.fillna(0, inplace=True)

        return _df_elems.astype(int)
    
    @staticmethod
    def get_key():
        key = ['no_ip',  'no_cas', 'no_cate', 'no_plan', 'tymouv', 'pe_chai_ddv']
        key = [x.upper() for x in key]

        return key

#-------------------------------------------------------------------------
# build chain matrix
#-------------------------------------------------------------------------
def build_chain_matrix(keys, groups, cm_ref):
    chain_matrix = pd.DataFrame()
     
    for k in keys  :
        cv = chain_vector(groups.get_group(k))
        cm = cv._to_elems_matrix(cm_ref)
        chain_matrix = pd.concat([chain_matrix, cm])

    return chain_matrix

#-------------------------------------------------------------------------

df_big_chain = helper_load_df_from_db(dir, 'chain', 'chain')

#-------------------------------------------------------------------------

groups = df_big_chain.groupby(by=chain_vector.get_key())

# define reference
cv_ref = chain_vector(df_big_chain)
cm_ref = cv_ref._to_elems_matrix(None)

#-------------------------------------------------------------------------
# operate on a sample
#-------------------------------------------------------------------------
ser_groups = pd.Series([k for k, g in groups])

ser_groups_sample = ser_groups

# ser_groups_sample = ser_groups.sample(1000)

# retrieve list of ip from sample

def tuple_to_str():
    def _f(tuple):
        s = ""
        for t in tuple:
            s = s + str(t)
            s = s + '_'
        return s
    return _f

f_tuple = tuple_to_str()

y = ser_groups_sample.apply(f_tuple).values
df_matrix = build_chain_matrix(ser_groups_sample, groups, cm_ref)
X = df_matrix.values

#-------------------------------------------------------------------------
# A. grid search / no TEST set, i.e test size  = 0
#-------------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0, random_state=0)

pipe = Pipeline([
    ('scaler', None), # Optional step
    ('knn', KNeighborsClassifier())
])

k_values = np.arange(1,7)
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
    params_dict['accuracy'] = pipe.score(X_tr, y_tr)
    # params_dict['accuracy'] = pipe.score(X_te, y_te)
     
    # Save result
    test_scores.append(params_dict)

df_scores = pd.DataFrame(test_scores).sort_values(by ='accuracy', ascending=False)
best_params = df_scores.iloc[0,:][1:]

pipe.set_params(**best_params)
pipe.fit(X_tr, y_tr)

cv_1 = chain_vector(groups.get_group((4250,4,0,1,'1','01.01.2017')))
cm_1 = cv_1._to_elems_matrix(cm_ref)

pipe.predict(cm_1)

#-------------------------------------------------------------------------
# B. Grid search cross validation
#-------------------------------------------------------------------------

grid_cv = GridSearchCV(pipe, [{
    'knn__n_neighbors': np.arange(1, 8)
}], cv=2, n_jobs=-1)

grid_cv.fit(X, y)
grid_cv.best_score_

#-------------------------------------------------------------------------
# make predictions
#-------------------------------------------------------------------------

cv_1 = chain_vector(groups.get_group((4250,1,0,1,'1','01.01.2019')))
cm_1 = cv_1._to_elems_matrix(cm_ref)

pred_ip = grid_cv.predict(cm_1)
idx_ip = y.tolist().index(pred_ip)
k = ser_groups_sample.iloc[idx_ip]
k

cv_1.compute_dist(chain_vector(groups.get_group(k)))

#-------------------------------------------------------------------------
# C. Empiric  
# Evaluates all the distances and take the closest
#-------------------------------------------------------------------------
results = []
for k, v in groups:
    cv = chain_vector(groups.get_group(k))
    results.append(cv.compute_dist(cv_1))

idx_k = pd.Series(results).idxmin()
k = ser_groups.iloc[idx_k]

#-------------------------------------------------------------------------
# D. NearestNeighbor  
# 
#-------------------------------------------------------------------------
n = NearestNeighbors(n_neighbors=40)
n.fit(X)

cv_1 = chain_vector(groups.get_group((4250,1,1,1,'1','01.01.2019')))
cm_1 = cv_1._to_elems_matrix(cm_ref)

a_dist, a_idx = n.kneighbors(cm_1)
l_groups = [ser_groups_sample.iloc[idx] for idx in a_idx[0]]

for group in l_groups:
    cv = chain_vector(groups.get_group(group))
    print (group, cv.compute_dist(cv_1))
    
neighbors_matrix = build_chain_matrix(l_groups, groups, cm_ref)

results = []
for i in np.arange(1,len(neighbors_matrix)):
    results.append(np.vdot(cm_1, neighbors_matrix.iloc[i,:]))
