# import mysql.connector
import os
import sqlite3
# import cx_Oracle    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time

#soldel_db = mysql.connector.connect(
#  host="localhost",
#  user="root",
#  passwd="waldo",
#  database="mupe"
#)

# connection = cx_Oracle.connect("Data Source = LABCIT; User ID = PADEV96_DATA; Password = PADEV96_DATA")

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
data_file = os.path.join(dir, 'chain.xlsx')

def read_and_store(data_file):
    df_chain = pd.read_excel(data_file)
    # df_chain = pd.DataFrame.from_csv(data_file, sep='\t')
  
    df_chain['PE_CHAI_DDV'] = df_chain['PE_CHAI_DDV'].apply(f_format_date)
    
    # df_chain['PE_CHAI_DDV'] = df_chain['PE_CHAI_DDV'].str.replace(r' 00:00:00', "")
    helper_store_df_to_db(df_chain, dir, 'chain', 'chain')


if 0:
    data_file = os.path.join(dir, 'chain.xlsx')
    read_and_store(data_file)

    data_file = os.path.join(dir, 'chain_1.xlsx')
    read_and_store(data_file)

#-------------------------------------------------------------------------

df_chain = helper_load_df_from_db(dir, 'chain', 'chain')

#-------------------------------------------------------------------------
class chain_vector:
    _df_items = None

    def __init__(self, df):
        self._df_items = df.copy()[['NOM_ELEM', 'NOM_LOGI']]
        self._df_items.drop_duplicates(inplace=True)

    def compute_dist(self, other):
        if other == self:
            return -1

        if other != None:
            lunion = len(pd.merge(self._df_items, other._df_items, how='outer'))
            lintersection = len(pd.merge(self._df_items, other._df_items, how='inner'))
            
            return  1 - lintersection / lunion
        
# key of chain vector
key_vector = ['no_ip',  'no_cas', 'no_cate', 'no_plan', 'tymouv', 'pe_chai_ddv']
key_vector = [x.upper() for x in key_vector]

gr = df_chain.groupby(by=key_vector)

# UNIT TEST

chain_1 = chain_vector(gr.get_group((5750,1,1,1,'1','01.01.2005')))
chain_2 = chain_vector(gr.get_group((4250,1,1,1,'1','01.01.2005')))
chain_1.compute_dist(chain_2)

# reference which contais all different defined elements 
chain_ref = chain_vector(df_chain)

results = []
for key_vector, g in gr:
    result = {}
 
    chain = chain_vector(g)
    result['dist_to_ref'] = chain.compute_dist(chain_ref)
    result['key_vector'] = key_vector
    results.append(result)

df_results = pd.DataFrame(results)
df_results.sort_values(by = 'reference_dist', ascending=False, inplace=True)
df_results.index = np.arange(len(df_results))
df_results.hist()
plt.show()

