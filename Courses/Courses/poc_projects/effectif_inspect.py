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
# digression HSBC
# 
#-------------------------------------------------------------------------
def produce_df(filename):
    df = pd.read_excel(os.path.join(dir, str(filename) + ".xlsx" ))
    ser_index = pd.Series(df.columns)

    # une solution pour retirer tout ce qui n'est pas alnum (date par exemple)
    ser_index = ser_index[~ser_index.str.isalnum().isnull()]
    ser_index.sort_values(inplace=True)

    df = df.reindex(columns=ser_index)
    df.set_index(keys=['NPERSO'],inplace=True)
    df.to_csv(os.path.join(dir, str(filename) + ".csv"))
    return df    
    
filename = 'HSBC_before_corr_bonif'
if 0:
    produce_df(filename)

df_hsbc_before = pd.read_csv(os.path.join(dir, str(filename) + ".csv"))
df_hsbc_before.set_index(keys=['NPERSO'],inplace=True)

filename = 'HSBC_after_corr_bonif'
if 0:
    produce_df(filename)
    
df_hsbc_after = pd.read_csv(os.path.join(dir, str(filename) + ".csv"))
df_hsbc_after.set_index(keys=['NPERSO'],inplace=True)

#-------------------------------------------------------------------------
# main idea is to define subroup to compare 
#-------------------------------------------------------------------------
i_elems = df_hsbc_after.columns[df_hsbc_after.columns.str.contains('CEP')]

diff = df_hsbc_after[i_elems] - df_hsbc_before[i_elems]
diff.describe()

# identifies outliers
diff.idxmax()

# removes them
diff.drop(index=diff.idxmax().values, inplace=True)
diff.describe()


