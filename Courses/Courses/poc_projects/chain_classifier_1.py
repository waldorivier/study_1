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

# from natsort import natsorted

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
    _key = None

    def __init__(self, key, df):
        self._key = key
        self._df_items = df.copy()[['NOM_ELEM', 'NOM_LOGI']]
        self._df_items.drop_duplicates(inplace=True)
        self._build_logi_name()

    def _build_logi_name(self):
        self._df_items['ID_LOGI'] = self._df_items['NOM_ELEM'] + "_" + self._df_items['NOM_LOGI']

    def compute_dist(self, other):
        if other == self:
            return -1

        if other != None:
            lunion = len(pd.merge(self._df_items, other._df_items, how='outer'))
            lintersection = len(pd.merge(self._df_items, other._df_items, how='inner'))
            
            # return  1 - lintersection / lunion
            return lintersection
    
    def _to_elems_matrix(self, cm_ref):
        assert self._df_items['ID_LOGI'] is not None
        
        ser_elems = self._df_items['ID_LOGI']
        ser_elems = ser_elems.append(pd.Series('KEY'))

        ones = np.ones((1, ser_elems.shape[0]))
        df_elems = pd.DataFrame(ones, columns=ser_elems)
        df_elems.astype(int)

        df_elems['KEY'] = [tuple(self._key)]
        
        if cm_ref is not None:
            df_elems = df_elems.reindex(columns=cm_ref.columns)
            df_elems.fillna(0, inplace=True)

        return df_elems
    
    @staticmethod
    def get_key_attribs():
        l_attribs = ['no_ip',  'no_cas', 'no_cate', 'no_plan', 'tymouv', 'pe_chai_ddv']
        l_attribs = [x.upper() for x in l_attribs]

        return l_attribs

#-------------------------------------------------------------------------
# build chain matrix
#-------------------------------------------------------------------------
def build_chain_matrix(keys, groups, cm_ref):
    chain_matrix = pd.DataFrame()
     
    for k in keys  :
        cv = chain_vector(k, groups.get_group(k))
        cm = cv._to_elems_matrix(cm_ref)
        chain_matrix = pd.concat([chain_matrix, cm])

    return chain_matrix

#-------------------------------------------------------------------------
#  contient tous les éléments mais ceux ci ne sont pas groupés
#-------------------------------------------------------------------------

df_big_chain = helper_load_df_from_db(dir, 'chain', 'chain')

#-------------------------------------------------------------------------

gb_keys = df_big_chain.groupby(by=chain_vector.get_key_attribs())

# define reference
cv_ref = chain_vector("REF", df_big_chain)
cm_ref = cv_ref._to_elems_matrix(None)

#-------------------------------------------------------------------------
ser_keys = pd.Series([k for k, g in gb_keys])

def tuple_to_str():
    def _f(tuple):
        s = ""
        for t in tuple:
            s = s + str(t)
            s = s + '_'
        return s
    return _f

f_tuple = tuple_to_str()

if 0:
    # produit la matrice des éléments groupés par chainage
   
    df_matrix = build_chain_matrix(ser_keys, gb_keys, cm_ref)
    df_matrix.to_csv(os.path.join(dir, 'big_chain_matrix.csv'), index=False)

#-------------------------------------------------------------------------
# A. grid search / no TEST set, i.e test size  = 0
#-------------------------------------------------------------------------

if 0:
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

if 0:
    grid_cv = GridSearchCV(pipe, [{
        'knn__n_neighbors': np.arange(1, 8)
    }], cv=2, n_jobs=-1)

    grid_cv.fit(X, y)
    grid_cv.best_score_

    #-------------------------------------------------------------------------
    # make predictions
    #-------------------------------------------------------------------------

    cv_1 = chain_vector(groups.get_group((4250,1,1,1,'1','01.01.2019')))
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

if 0:
    results = []
    for k, v in groups:
        cv = chain_vector(groups.get_group(k))
        results.append(cv.compute_dist(cv_1))

    idx_k = pd.Series(results).idxmin()
    k = ser_groups.iloc[idx_k]

#-------------------------------------------------------------------------
# D. NearestNeighbors  
#-------------------------------------------------------------------------

def find_neighbors_of(df_matrix, gb_keys, k_target, cond):
    class res:
        _KEY = None
        _dist = None
        def __init__(self, KEY, dist):
            self._KEY = KEY
            self._dist = dist

        def as_dict(self):
            return {'KEY' : self._KEY,
                    'dist': self._dist}

    i_elems = df_matrix.columns[df_matrix.columns.str.contains(cond)]

    # dépend la condition
    df_matrix_f = df_matrix[df_matrix[i_elems].sum(axis=1)== 2]
    df_matrix_f.index = np.arange(len(df_matrix_f))

    X = df_matrix_f.drop(columns=['KEY']).values
    n = NearestNeighbors(n_neighbors=40)
    n.fit(X)

    cv_1 = chain_vector(target_k, gb_keys.get_group(target_k))
    cm_1 = cv_1._to_elems_matrix(cm_ref).drop(columns=['KEY'])

    a_dist, a_idx = n.kneighbors(cm_1)
    l_keys = [eval(df_matrix_f.loc[idx,'KEY']) for idx in a_idx[0]]

    l_neighbors = []
    for k in l_keys:
        #  chains of k will be skiped
        if k[0] != target_k[0]: 
            cv = chain_vector(k, gb_keys.get_group(k))
            l_neighbors.append(res(k, cv.compute_dist(cv_1)))
    
    return pd.DataFrame([x.as_dict() for x in l_neighbors])

#-------------------------------------------------------------------------

df_matrix = pd.read_csv(os.path.join(dir, 'big_chain_matrix.csv'))

cond = 'CFAASX|CRISCA'
target_k = (3290,1,1,1,'1','31.12.2017')
df_neighbors = find_neighbors_of(df_matrix, gb_keys, target_k, cond)

optimal_k = df_neighbors.iloc[0,0]
optimal_items = chain_vector(optimal_k,gb_keys.get_group(optimal_k))._df_items
logi_cond = optimal_items.ID_LOGI.str.contains(cond)
optimal_items[logi_cond]


