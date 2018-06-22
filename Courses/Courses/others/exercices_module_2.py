import os

from pathlib import PureWindowsPath
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import requests as r

from bs4 import BeautifulSoup

#-------------------------------------------------------------------------
# lecture du fichier contenant les données 

#working_dir = PureWindowsPath(os.getcwd())
#data_dir = PureWindowsPath(working_dir.joinpath('Data'))
#data_file = PureWindowsPath(data_dir.joinpath('module_2_data_1.xls'))
#df = pd.read_excel(data_file.as_posix())

# lecture des deux premières colonnes
# df = pd.read_excel(data_file.as_posix(), sheet_name='Sheet2',usecols=[0,1])

page = r.get('http://time.com/4572079/best-inventions-2016/')
soup = BeautifulSoup(page.text, 'html.parser')