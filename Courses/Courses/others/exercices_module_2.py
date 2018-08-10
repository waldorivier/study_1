import os
import pandas

from pathlib import PureWindowsPath

import numpy as np
import pandas as pd
import requests as r

import matplotlib.pyplot as plt
import seaborn

from bs4 import BeautifulSoup
import sqlite3

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

    # supprime les colonnes (axis = 1) qui on au moins plus 4000 lignes manquantes
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

if 0 : 
    ser_date = pd.date_range(pd.Timestamp('2010-01-01'), periods=10, freq='M')
    for d in ser_date:
        print(d.weekday_name)

    data_file = PureWindowsPath(data_dir.joinpath('financial_data.csv'))
    data = pd.read_csv(data_file, index_col='Date')
    # data.set_index('Date')
    data.index = pd.to_datetime(data.index)
    
    # data.plot(figsize=(16,12), style='-')
    # Différences entre asfrequ et resampling qui elle, applique une fonction (=moyenne)
   
    # data.resample('BA').mean().plot(style=':')
    # data.asfreq('BA').plot(style='--')
    # plt.legend(['input', 'resample', 'asfreq'], loc='upper left');
    # plt.show()

    rolling = data.rolling(365, center=True)
    data.plot(figsize=(16,12))
    rolling.mean().plot(color='red', linewidth=3)
    plt.show()

#-------------------------------------------------------------------------
# Databases
#-------------------------------------------------------------------------

if 0 :
    # helper class pour exécuter un query via un dataframe
    def run_query(query):
        return pd.read_sql_query(query,db)

    # définition d'une base de données
    db = sqlite3.connect('consumer.db')
 
    data_file = PureWindowsPath(data_dir.joinpath('consumer_complaints.csv'))

    # lecture par bloc : chunksize correspond au nombre de lignes lues par bloc

    for chunk in pd.read_csv(data_file, chunksize=2000):
        chunk.to_sql(name="data", con=db, if_exists="append", index=False)  

    run_query("SELECT tbl_name FROM sqlite_master;")

    # variante utilisant un curseur
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    # restitution du résultat
    results = cursor.fetchall()

#-------------------------------------------------------------------------
# Databases / données du projet module 2
# le fichier est en format tsv (tab separator) et contient plus de 
# 1'300'000 lignes
#-------------------------------------------------------------------------

if 1 :    
    #-------------------------------------------------------------------------
    # déclaration d'une base de données temporaire pour le chargement 
    # d'un DataFrame
    #-------------------------------------------------------------------------

    db_file = PureWindowsPath(data_dir.joinpath('food.db'))
    db = sqlite3.connect(db_file.as_posix())
  
    # data_file = PureWindowsPath(data_dir.joinpath('en.openfoodfacts.org.products.tsv'))
 
    # load en base de données par bloc pour éviter un memory overflow (load mis en commentaire)
    # for chunk in pd.read_csv(data_file.as_posix(), delimiter='\t', encoding='utf-8', chunksize=2000):
    #    chunk.to_sql(name="data", con=db, if_exists="append", index=False)  
    
    #-------------------------------------------------------------------------
    # load du df à partir de la base de données
    #-------------------------------------------------------------------------

    df_food = pd.read_sql_query("select * from data", con=db)
    
    #-------------------------------------------------------------------------
    # nettoyage des données
    # suppression de toutes les colonnes qui n'ont pas de données
    # réduction du tableau à 147 colonnes (initialement 163)
    #-------------------------------------------------------------------------

    df_food.dropna(axis=1, how="all", inplace=True)

    #-------------------------------------------------------------------------
    # exporter les colonnes avec le nombre de valeurs nulles afin de 
    # visualiser les caractéristiques qui ne sont pas relevantes / ou qui ne
    # nous interesent tout simplement pas 
    #-------------------------------------------------------------------------

    ser_column_null = df_food.isnull().sum()
    ser_column_null.to_csv(data_dir.joinpath(data_result_name))

    #-------------------------------------------------------------------------
    # gérer les doublons 
    # en l'occurence, pour les colonnes selectionnées, il n'y a pas de doublon
    #-------------------------------------------------------------------------
    
    if df_food.drop_duplicates().shape == df_food.shape :
        print ("le tableau ne contient aucune ligne dupliquée")
    
    #-------------------------------------------------------------------------
    # suppression de toutes les colonnes qui contiennent un nombre 
    # supérieur ou égal à 1'000'000 de valeurs nulles 
    # ou une autre dimension permettant une optimisation des caractéristiques 
    # conservées minimisant les valeurs nulles résiduelles. 
    #
    # La motivation est d'extraire un sous-ensemble le plus large que possible
    # en terme de valeurs non nulles
    #-------------------------------------------------------------------------

    threshes = np.arange(0, df_food.isnull().sum().max(), 10000)

    # construction d'un DF à partir d'une liste de row construite 
    # à l'aide d'un dictionnaire
    rows = []

    def optimize_col_selection(thresh):
        df = df_food.dropna(thresh=thresh, axis=1)
        
        dict = {'thresh_' : 0, 'schape_':0, 'mean_':0}
        dict['thresh_'] = thresh
        dict['schape_'] = df.shape[1]
        dict['mean_']   = df.isnull().sum().mean()

        rows.append(dict)
        
    for thresh in threshes :
        optimize_col_selection (thresh)
    
    df_result = pd.DataFrame(rows)  
    df_result.set_index('schape_', inplace=True)
    
    #-------------------------------------------------------------------------
    # le tableau ci-dessous indique le nombre de critères conservés et 
    # la moyenne des valeurs nulles des colonnes résiduelles conservées
    # par niveau (thresh) en fonction duquel une colonne est supprimée.   
    #-------------------------------------------------------------------------
    df_result

    df_result.plot(figsize=(16,12))
    plt.xlabel('nombre de colonnes')
    plt.legend(labels=['moyenne du nombre de valeur nulle','niveau'])

    plt.show()

    #-------------------------------------------------------------------------
    # le niveau de 280'000 définit une sélection de 25 critères (comprenant 
    # "ingredients_text" nécessaire à notre analyse)
    #-------------------------------------------------------------------------
    df_food_sel = df_food.dropna(thresh=280000, axis=1)
    
    #-------------------------------------------------------------------------
    # autorise l'affichage de toutes les colonnes
    #-------------------------------------------------------------------------
    pd.set_option('display.max_columns', df.shape[1]) 
    df_food_sel.head()

    #-------------------------------------------------------------------------
    # suppression de toute les colonnes qui contiennent les mots 
    # to-be-completed
    #     
    # par exemple la colonne "states_tags" contient en grand nombre le texte "to-be-completed"
    # qui n'est pas relevant pour notre étude
    #-------------------------------------------------------------------------

    df_food_sel.states_tags.where(lambda x : x.str.contains("to-be-completed")).count()
    
    # on décide de supprimer toutes les colonnes dont le nom contient "states"


