
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
    df = pd.read_csv(dataFile.as_posix())

    df.info()
    df.columns()
    df.dtypes

    df_numeric_columns = df.select_dtypes(include=['float64'])

    numTypes = df.dtypes == (np.float64)
    for i in range(numTypes.count()) :
        print (numTypes[i])
        print (df.columns[i])
        if numTypes[i] == True :
            sum = df[df.columns[i]].sum()
            print (sum)

except ValueError as e  :
    print (e)
       