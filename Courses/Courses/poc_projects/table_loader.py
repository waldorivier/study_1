import os
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse

import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------
# répertoire de travail
#-------------------------------------------------------------------------

working_dir = PureWindowsPath(os.getcwd())
data_dir = PureWindowsPath(working_dir.joinpath('poc_projects'))
data_file = PureWindowsPath(data_dir.joinpath('table_data.xlsx'))

#-------------------------------------------------------------------------

tbl_columns = ['NOM_PARA', 'PE_PAUT_DDV', 'LIBF_PARA', 'TYSTRU_PARA', 
               'LCRD1F_PARA','TYCRD1_PARA', 
               'LCRD2F_PARA','TYCRD2_PARA',
               'LCRD3F_PARA','TYCRD3_PARA',
               'TYDATA_PARA','FORMAT_PARA',
               'CLATIT_PAUT','NO_IP','NO_PLAN','NO_CAS',
               'COORD1_VAPA_MF','COORD2_VAPA_MF','COORD3_VAPA_MF','VALSTR_VAPA','INCLCOLLID','COLLECTIF','VALNUM_VAPA']

# define a table template

rows = []
row = {}
row['NOM_PARA']     = "AX_AK" 
row['PE_PAUT_DDV']  = "01.01.2019" 
row['LIBF_PARA']    = "Expectative enfant"
row['TYSTRU_PARA']  = "02"

row['CLATIT_PAUT']   = "PE_IP"
row['NO_IP']         = 4250
row['NO_PLAN']       = 1
row['NO_CAS']        = 3

# load data from XL

df_tbl_data = pd.read_excel(data_file)
tbl_data_colums = df_tbl_data.columns

i = 1
for c in tbl_data_colums[:-1]:
    col = "LCRD" + str(i) + "F_PARA"
    row[col] = c
    i += 1

row_template = row;
rows.append(row_template)

for index, row in df_tbl_data.iterrows():
    _row = row_template
    
    i = 1
    for c in tbl_data_colums[:-1]:
        col = "LCRD" + str(i) + "F_PARA"
        _row[col] = row[i]
        i += 1

    rows.append(_row)

df_tbl = pd.DataFrame(rows, columns = tbl_columns)
df_tbl.fillna(method='ffill', inplace=True)

df_tbl.to_csv('table.csv', sep = ';', index=False)

