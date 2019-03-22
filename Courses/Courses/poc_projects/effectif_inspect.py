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

dir = os.getcwd()
dir = os.path.join(dir, 'poc_projects')
   
#-------------------------------------------------------------------------
# digression 
# 
#-------------------------------------------------------------------------
def produce_df(filename):
    df = pd.read_excel(os.path.join(dir, str(filename) + ".xlsx" ))
    # ser_index = pd.Series(df.columns)

    # une solution pour retirer tout ce qui n'est pas alnum (date par exemple)
    # ser_index = ser_index[~ser_index.str.isalnum().isnull()]
    # ser_index.sort_values(inplace=True)

    # df = df.reindex(columns=ser_index)
    # df.set_index(keys=['NPERSO'],inplace=True)
    df.set_index(keys=['NEMPLO'],inplace=True)
    df.to_csv(os.path.join(dir, str(filename) + ".csv"))
    return df    
    
filename = 'fact_mikron_after_3'
produce_df(filename)
df_fact_after = pd.read_csv(os.path.join(dir, str(filename) + ".csv"))
df_fact_after.set_index(keys=['NEMPLO'],inplace=True)

filename = 'fact_mikron_before'
produce_df(filename)
df_fact_before = pd.read_csv(os.path.join(dir, str(filename) + ".csv"))
df_fact_before.set_index(keys=['NEMPLO'],inplace=True)

cols = df_fact_before.columns[12:]
for c in cols:
    print (c, df_fact_before[c].sum())

cols = df_fact_after.columns[12:]
for c in cols:
    print (c, df_fact_after[c].sum())


cols = df_fact_before.columns[12:]
for c in cols:
    data_before = df_fact_after.groupby(['NEMPLO'])[c].sum()
    data_after = df_fact_before.groupby(['NEMPLO'])[c].sum()

    print (c, data_after[data_after - data_before != 0])