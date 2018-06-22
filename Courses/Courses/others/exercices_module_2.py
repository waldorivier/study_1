import os

from pathlib import PureWindowsPath
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

#-------------------------------------------------------------------------
# lecture du fichier contenant les données 

working_dir = PureWindowsPath(os.getcwd())
data_dir = PureWindowsPath(working_dir.joinpath('Data'))
data_file = PureWindowsPath(data_dir.joinpath('module_2_data_1.xls'))
df = pd.read_excel(data_file.as_posix())

# lecture des deux premières colonnes
df = pd.read_excel(data_file.as_posix(), sheet_name='Sheet2',usecols=[0,1])