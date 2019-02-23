#import mysql.connector
import os
import sqlite3
import cx_Oracle    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

soldel_db = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="waldo",
  database="mupe"
)

connection = cx_Oracle.connect("Data Source = LABCIT; User ID = PADEV96_DATA; Password = PADEV96_DATA")

#-------------------------------------------------------------------------
# persiste / charge un DataFrame vers / de la  base de données 
# REM : db_name sans extension ; le nom de la table correspond à celui de 
#       la base de données
#-------------------------------------------------------------------------
def helper_store_df_to_db(df, dir, db_name, table_name) :
    db_file = os.path.join(dir, db_name + ".db" )
    db = sqlite3.connect(db_file)

    df.to_sql(table_name, db, chunksize=2000, if_exists='replace')
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

dir = os.getcwd()
dir = os.path.join(dir, 'poc_projects')
data_file = os.path.join(dir, 'chain.xlsx')

if 0:
    df_chain = pd.read_excel(data_file)
    helper_store_df_to_db(df_chain, dir, 'chain', 'chain')

#-------------------------------------------------------------------------

df_chain = helper_load_df_from_db(dir, 'chain', 'chain')

#-------------------------------------------------------------------------
class chain_vector:
    _ser_items = None
    _nb_items = None
    _ref_dist = None

    def __init__(self, df):
        self._ser_items = df.copy()[['NOM_ELEM', 'NOM_LOGI']]
        self._ser_items.drop_duplicates(inplace=True)
        self._nb_items = len(self._ser_items)

    def compute_ref_dist(self, nb_ref_items = None):
        if nb_ref_items == 0 or nb_ref_items is None:
            self._ref_dist = self._nb_items
        else:
            self._ref_dist = self._nb_items / nb_ref_items

        return self._ref_dist

    def compute_dist(self, other):
        # 
        if other == self:
            return -1

        if self._ref_dist != None:
            return np.abs(other._ref_dist - self._ref_dist)
        else :
            raise ValueError('_ref_dist must be set')    

# key of chain vector
key_vector = ['no_ip',  'no_cas', 'no_cate', 'tymouv', 'pe_chai_ddv', 'no_plan',]
key_vector = [x.upper() for x in key_vector]

ref = chain_vector(df_chain)
ref.compute_ref_dist()

gr = df_chain.groupby(by=key_vector)

results = []
for key_vector, g in gr:
    result = {}
 
    chain = chain_vector(g)
    result['reference_dist'] = chain.compute_ref_dist(ref._ref_dist)
    result['key_vector'] = key_vector
    
    results.append(result)

df_results = pd.DataFrame(results)
df_results.sort_values(by = 'reference_dist', ascending=False, inplace=True)
df_results.index = np.arange(len(df_results))
df_results.hist()
plt.show()

