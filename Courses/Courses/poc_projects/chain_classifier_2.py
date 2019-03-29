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
from sklearn.cluster import KMeans

# Create k-means object
kmeans = KMeans(
    n_clusters=3,
    random_state=0 # Fix results
)

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

#-------------------------------------------------------------------------

def read_and_store(data_file):
    df_chain = pd.read_excel(data_file)
    # df_chain = pd.DataFrame.from_csv(data_file, sep='\t')
  
    df_chain['PE_CHAI_DDV'] = df_chain['PE_CHAI_DDV'].apply(f_format_date)
    
    # df_chain['PE_CHAI_DDV'] = df_chain['PE_CHAI_DDV'].str.replace(r' 00:00:00', "")
    helper_store_df_to_db(df_chain, dir, 'chain', 'chain')

if 0:
    data_file = os.path.join(dir, 'chain.xlsx')
    read_and_store(data_file)

#-------------------------------------------------------------------------
# build chain matrix (ELEM_1, ELEM_2, ..., ELEM_N)
# each row identified by KEY as tuple (==chain key) contains 1 if ELEM i belongs
# to chain, else 0
#-------------------------------------------------------------------------
def build_chain_matrix(gb_keys, cm_ref):
    chain_matrix = pd.DataFrame()
     
    for k, v in gb_keys :
        cv = chain_vector(k, v)
        cm = cv._to_elems_matrix(cm_ref)
        chain_matrix = pd.concat([chain_matrix, cm])

    return chain_matrix

#-------------------------------------------------------------------------
# represents a chain as a vector of elems ELEM, LOGI
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
        self._df_items.set_index(keys=['ID_LOGI'], inplace=True)

    def compute_dist(self, other):
        if other == self:
            return -1

        if other != None:
            lunion = len(pd.merge(self._df_items, other._df_items, how='outer'))
            lintersection = len(pd.merge(self._df_items, other._df_items, how='inner'))
            
            # return  1 - lintersection / lunion
            return lintersection
    
    def _to_elems_matrix(self, cm_ref):
        idx_elems = self._df_items.index
        idx_elems = idx_elems.append(pd.Index(['KEY']))

        ones = np.ones((1, idx_elems.shape[0]))
        df_elems = pd.DataFrame(ones, columns=idx_elems)
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

    # to use as lambda (function which accept lambda as parameter)
    @staticmethod
    def _tuple_to_str():
        def _f(tuple):
            s = ""
            for t in tuple:
                s = s + str(t)
                s = s + '_'
            return s
        return _f

#-------------------------------------------------------------------------

if 0:
    # produit la matrice des éléments groupés par chainage
   
    df_chain_matrix = build_chain_matrix(gb_keys, cm_ref)
    df_chain_matrix.to_csv(os.path.join(dir, 'chain_matrix.csv'), index=False)

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def find_neighbors_of(df_chain_matrix, gb_keys, cm_ref, target_k, cond):
    
    # inner class
    class res:
        _KEY = None
        _dist = None
        def __init__(self, KEY, dist):
            self._KEY = KEY
            self._dist = dist

        def as_dict(self):
            return {'KEY' : self._KEY,
                    'dist': self._dist}

    #---------------------------------------------------------------------

    idx_cols_cond = df_chain_matrix.columns[df_chain_matrix.columns.str.contains(cond)]

    # dépend la condition
    df_matrix_f = df_chain_matrix[df_chain_matrix[idx_cols_cond].sum(axis=1) >= 1]
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
# return all sibling chains (i.e belonging to the same cluster) of target_k chain
#-------------------------------------------------------------------------
def find_cluster_of(df_chain_matrix, gb_keys, cm_ref, target_k, cond):

    idx_cols_cond = df_chain_matrix.columns[df_chain_matrix.columns.str.contains(cond)]

    # depends on the condition
    df_matrix_f = df_chain_matrix[df_chain_matrix[idx_cols_cond].sum(axis=1) >= 1]
    df_matrix_f.index = np.arange(len(df_matrix_f))

    X = df_matrix_f.drop(columns=['KEY']).values
    n = KMeans(n_clusters=100)
    n.fit(X)
    a_labels = n.labels_

    cv_1 = chain_vector(target_k, gb_keys.get_group(target_k))
    cm_1 = cv_1._to_elems_matrix(cm_ref).drop(columns=['KEY'])

    a_cluster = n.predict(cm_1)

    return df_matrix_f[a_labels == a_cluster[0]]

#-------------------------------------------------------------------------
# LETS GO
#-------------------------------------------------------------------------

df_elems = helper_load_df_from_db(dir, 'chain', 'chain')
gb_keys = df_elems.groupby(by=chain_vector.get_key_attribs())

# define reference vector 

cv_ref = chain_vector("REF", df_elems)
cm_ref = cv_ref._to_elems_matrix(None)

#-------------------------------------------------------------------------
# chains already converted in matrix form

df_chain_matrix = pd.read_csv(os.path.join(dir, 'chain_matrix.csv'))
df_elne = pd.read_csv(os.path.join(dir, 'chain_elne.csv'))
df_elne.set_index(keys=['ID_LOGI'],inplace=True)

#-------------------------------------------------------------------------

cond = 'RVPRET'
target_k = (4400,1,1,1,'1','01.01.2018')

df_neighbors = find_neighbors_of(df_chain_matrix, gb_keys, cm_ref, target_k, cond)

optimal_k = df_neighbors.iloc[0,0]
optimal_k
optimal_items = chain_vector(optimal_k,gb_keys.get_group(optimal_k))._df_items
logi_cond = optimal_items.index.str.contains(cond)

pd.merge(optimal_items[logi_cond], df_elne)

#-------------------------------------------------------------------------
# CLUSTERING / comparison with 
#-------------------------------------------------------------------------

df_cluster = find_cluster_of(df_chain_matrix, gb_keys, cm_ref, target_k, cond)

optimal_k = (1910, 1, 1, 2, '1', '31.12.2013')  
optimal_chain = chain_vector(optimal_k,gb_keys.get_group(optimal_k))

intersect = optimal_chain.compute_dist(chain_vector(target_k,gb_keys.get_group(target_k)))
intersect                                                    

optimal_g = gb_keys.get_group(optimal_k) 

#-------------------------------------------------------------------------
# finds dependencies
#-------------------------------------------------------------------------

elem_base = 'CASS2X'
df_ = df_elne.loc[df_elne.index[df_elne.index.str.contains(elem_base)]]

#-------------------------------------------------------------------------
# variant with facturation's elem
# try to find a consensus 
#-------------------------------------------------------------------------

data_file = 'fact_elem.xlsx'
data_file = os.path.join(dir, data_file)
df_elem_fact = pd.read_excel(data_file)

features = ['NO_IP', 'NO_PLAN', 'NO_CAS', 'NO_CATE', 'NOM_ELEM_FAC', 'COMPTE_ELFA']
df_elem = df_elem_fact[features]
df_elem.set_index(['NO_IP', 'NO_PLAN','NO_CAS', 'NO_CATE', 'NOM_ELEM_FAC'], inplace=True)

gb = df_elem.groupby(['NOM_ELEM_FAC'])
gb.get_group('CRIAME').sort_values(by = ['COMPTE_ELFA'])


