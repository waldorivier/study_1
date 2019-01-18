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

client_root_dir = PureWindowsPath('O:\\')
dest_dir = PureWindowsPath(client_root_dir.joinpath('cap','24 BenAdmin-Param','01 Analyse règlement','Tabelles_2019'))

#-------------------------------------------------------------------------

tbl_columns = ['NOM_PARA', 'PE_PAUT_DDV', 'LIBF_PARA', 'TYSTRU_PARA', 
               'LCRD1F_PARA','TYCRD1_PARA', 
               'LCRD2F_PARA','TYCRD2_PARA',
               'LCRD3F_PARA','TYCRD3_PARA',
               'TYDATA_PARA','FORMAT_PARA',
               'CLATIT_PAUT','NO_IP','NO_PLAN','NO_CAS',
               'COORD1_VAPA_MF','COORD2_VAPA_MF','COORD3_VAPA_MF','VALSTR_VAPA','INCLCOLLID','COLLECTIF','VALNUM_VAPA', '']

# define a row template

def generate_file():
    row_template = {}
    row_template['NOM_PARA']     = "AX_IK" 
    row_template['PE_PAUT_DDV']  = "01.01.2019" 
    row_template['LIBF_PARA']    = "Expectative enfant"
    row_template['TYSTRU_PARA']  = "02"
    row_template['FORMAT_PARA']  = "06"
    row_template['INCLCOLLID']   = "02"

    row_template['CLATIT_PAUT']   = "PE_IP"
    row_template['NO_IP']         = 4250
    row_template['NO_PLAN']       = 1
    row_template['NO_CAS']        = 8

    # load data from XL

    df_tbl_data = pd.read_excel(data_file)
    tbl_data_colums = df_tbl_data.columns

    # header contains name
    i = 1
    for c in tbl_data_colums[:-1]:
        col = "LCRD" + str(i) + "F_PARA"
        row_template[col] = c
        i += 1

    # first line contains type
    # coordinate type 
    i = 1
    for c in df_tbl_data.iloc[0,:-1]:
        col = "TYCRD" + str(i) + "_PARA"
        row_template[col] = c
        i += 1

    # data type 
    col = 'TYDATA_PARA';
    row_template[col] = df_tbl_data.iloc[0,-1]

    df_tbl_data = df_tbl_data.iloc[1:,:]
    rows = []
    for index, row in df_tbl_data.iterrows():
        _row = row_template.copy()
    
        i = 1
        for c in tbl_data_colums[:-1]:
            col = "COORD" + str(i) + "_VAPA_MF"
            _row[col] = row[i-1]
            i += 1
    
        _row[tbl_data_colums[-1]] = row[-1]
        rows.append(_row)

    df_tbl = pd.DataFrame(rows, columns = tbl_columns)
    df_tbl.to_csv(dest_dir.joinpath(row_template['NOM_PARA'] + ".csv"), sep = ';', index=False)

generate_file()