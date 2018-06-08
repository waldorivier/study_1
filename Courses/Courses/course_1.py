
import os
from sqlalchemy.dialects.mssql.information_schema import columns
from pathlib import PureWindowsPath
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

try :

    working_dir = PureWindowsPath(os.getcwd())
    data_dir = PureWindowsPath(working_dir.joinpath('Data'))
    data_file = PureWindowsPath(data_dir.joinpath('data.csv'))
    df = pd.read_csv(data_file.as_posix())

    df.info()
    df.columns
    df.dtypes

    df.count()
    df.dropna(inplace=True)
    df.count()

    # selection des colonnes de type numérique don on produit une copie
    df_num_col = df.select_dtypes(include=['float64', 'int64']).copy()

    # produit une série somme / max / min des valeurs prises par colonne
    sum_num_col = df_num_col.sum()
    max_num_col = df_num_col.max()   
    min_num_col = df_num_col.min()   

    # les 10 pays les plus "heureux"
    df_hap_asc = df.sort_values(by = 'Happiness Rank', ascending = True)
    df_hap_asc.head(10)

    # les 10 pays les moins "malheureux"
    df_hap_desc = df.sort_values(by = 'Happiness Rank', ascending = False)
    df_hap_desc.head(10)




except ValueError as e  :
    print (e)
       