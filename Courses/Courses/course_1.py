
import os
from sqlalchemy.dialects.mssql.information_schema import columns
from pathlib import PureWindowsPath
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

try :

    workingDir = PureWindowsPath(os.getcwd())
    dataDir = PureWindowsPath(workingDir.joinpath('Data'))
    dataFile = PureWindowsPath(dataDir.joinpath('data.csv'))
    df = pd.read_csv(dataFile.as_posix())

    df.info()
    df.columns()
    df.dtypes

    numTypes = df.dtypes == (np.float64)
    for i in range(numTypes.count()) :
        # print (numTypes[i])
        # print (df.columns[i])
        if numTypes[i] :
            df[df.columns[i]].sum()

except ValueError as e  :
    print (e)
       