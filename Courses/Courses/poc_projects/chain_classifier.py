#import mysql.connector
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


class chain_vector:
    _ser_items = None
    _nb_items = None
    _ref_dist = None

    def __init__(self, df):
        self._ser_items = df.copy()[['nom_elem', 'nom_logi']]
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

df = pd.read_sql_query("select * from pe_elca", con=soldel_db)

ref = chain_vector(df)
ref.compute_ref_dist()

gr = df.groupby(by=key_vector)

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

