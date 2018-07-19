import os
import pandas

from pathlib import PureWindowsPath

import numpy as np
import pandas as pd
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
# cf dta_inspect pour des exemples de merge
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
# Text data / Expressions régulières
# Le multi-index doit encore être revu
#-------------------------------------------------------------------------
if 0:
    s2 = pd.Series(['bar', 'sugar', 'cartoon', 'argon'])
    s2.str.contains('.ar')

if 0:
    meal_plan = ['Monday: 9:12am – Omelet,  3:30pm– Apple slices with almond butter', 
             'Tuesday: 9:35am – Banana bread, 11:00am –Sauteed veggies, 7:02pm– Taco pie',
             'Wednesday: 9:00am – Banana pancakes',  
             'Thursday: 7:23pm– Slow cooker pulled pork', 'Friday: 3:30pm – Can of tuna', 
             'Saturday: 9:11am: Eggs and sweet potato hash browns, 3:22pm: Almonds', 
             'Sunday: 11:00am: Meat and veggie stir fry'] 

#-------------------------------------------------------------------------
# retourne un dataframe avec le split désiré
# ceci est un version un peu compliquée
#-------------------------------------------------------------------------

    def parse_day_line (day_line, day_of_week):
        part_day = day_line.partition(day_of_week + ':')
        part_day_plan = part_day[2];
        ser_day_plan = pd.Series(part_day_plan.split(','))
        df_day_plan = ser_day_plan.str.extractall('(\d+):(\d+)(\w{2}\W+)([\w|\s]+)')
        df_day_plan['DAY'] = day_of_week
        return df_day_plan

    df_meal_plan = pd.DataFrame(meal_plan, columns=['TEXT'])
    df_day_of_week = df_meal_plan.TEXT.str.extract('(\w+day)')

    i = 0
    for day in df_day_of_week.values :
        meal_plan_day = df_meal_plan.TEXT[i]
        if i == 0:
            df_result = parse_day_line(meal_plan_day, day)
        else: 
            df_result = pd.concat([df_result, parse_day_line(meal_plan_day, day)])
        i += 1

    # renommer les colonnes
    df_result.rename(columns={0:'heure', 1:'minute', 2:'meridium', 3:'repas'}, inplace=True)

    # formatter la colonne meridium 
    df_result.meridium = df_result.meridium.str.extract('([am|pm]{2})')

    # poser un index multiple (une colonne devient une colonne d'index)
    df_result.set_index(['DAY', 'heure', 'meridium'], inplace=True)
    df_result
   

#-------------------------------------------------------------------------
# Time SERIES
#-------------------------------------------------------------------------

ser_date = pd.date_range(pd.Timestamp('2010-01-01'), periods=10, freq='M')
for d in ser_date:
    print(d.weekday_name)