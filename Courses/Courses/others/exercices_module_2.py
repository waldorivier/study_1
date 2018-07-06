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
data_result_name = 'result.csv'

# lecture des deux premières colonnes de la seconde feuille de calcul du fichier excel 

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

if 0 :
    data_file = PureWindowsPath(data_dir.joinpath('tmdb_5000_movies.csv'))
    df_movies = pd.read_csv(data_file.as_posix())

    # nb valeurs manquantes sur toutes les colonnes et toutes les lignes
    df_movies.isnull().sum().sum()


    # supprimer seulement les lignes qui contiennent toutes des valeurs manquantes
    df_movies.isnull().dropna(how='all')

    # idem que ci-dessus mais pour les colonnes
    df_movies.dropna(how='all', axis=1)

    # supprime les colonnes (axis = 1) qui on moins plus 4000 lignes manquantes
    df_movies.dropna(thresh=4000, axis=1, inplace=True)

    # remplacement des données manquantes

    df_movies['overview'].fillna(value = 'OVERVIEW_MISSING', inplace=True)

    # sauvegarde
    df_movies.to_csv(PureWindowsPath(data_dir.joinpath(data_result_name)), sep=';')

    #  identification d'une ligne avec valeur manquante
    df_movies['release_date'][df_movies['release_date'].isnull()]
    df_movies['release_date'][4552:4555]

    # modification en avant ou en arrière bfill: ie la valeur précèdente / suivante vient remplacer la valeur manquante
    df_movies['release_date'].fillna(method='ffill')[4552:4555]

    # ici on remplace les valeurs manquantes par la moyenne de la colonne
    df_movies['runtime'].fillna(value=df_movies['runtime'].mean(), inplace=True)


#-------------------------------------------------------------------------
# Duplicates
#-------------------------------------------------------------------------
 
if 0 :
    df = pd.DataFrame({ 'color': ['blue','blue','red','red','blue'], 'value': [2,1,3,3,2]})

    # cette fconction retourne True pour toutes les lignes qui sont à double
    # ici, les dernières occurences sont conservées 
    # (donc les premières occ. sont à True == Duplicated)

    df.duplicated(keep='last')

    # retourne les doublons
    df.loc[df.duplicated(),:]

    # retourne toutes les valeurs sans les doublons
    df.drop_duplicates()

#-------------------------------------------------------------------------
# Jointures 
#-------------------------------------------------------------------------