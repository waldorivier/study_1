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

working_dir = PureWindowsPath(os.getcwd())
data_dir = PureWindowsPath(working_dir.joinpath('Data'))
data_file = PureWindowsPath(data_dir.joinpath('module_2_data_1.xls'))

# lecture des deux premières colonnes

if 0:
    df = pd.read_excel(data_file.as_posix())
    df = pd.read_excel(data_file.as_posix(), sheet_name='Sheet2',usecols=[0,1])

#-------------------------------------------------------------------------
# web scrapping
# exraction du text contenu dans les balises h2
#-------------------------------------------------------------------------

if 0:
    page = r.get('http://time.com/4572079/best-inventions-2016/')
    soup = BeautifulSoup(page.text, 'html.parser')

    scrapted_titles = []

    titles = soup.find_all('h2')
    for title in titles :
        scrapted_titles.append(title.text.strip())

    df_titles = pd.DataFrame(scrapted_titles, columns=['Invention'])

#-------------------------------------------------------------------------
# Movies cleaning / filtering
#-------------------------------------------------------------------------

data_file = PureWindowsPath(data_dir.joinpath('tmdb_5000_movies.csv'))
df_movies = pd.read_csv(data_file.as_posix())