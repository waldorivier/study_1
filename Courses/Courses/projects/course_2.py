import os
import pandas

from pathlib import PureWindowsPath

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn

import sqlite3

#-------------------------------------------------------------------------
# persiste / charge un DataFrame vers / de la  base de données 
# REM : db_name sans extension ; le nom de la table correspond à celui de 
#       la base de données
#-------------------------------------------------------------------------
def helper_store_df_to_db(df, db_name, table_name) :

    db_file = PureWindowsPath(data_dir.joinpath(db_name + ".db" ))
    db = sqlite3.connect(db_file.as_posix())

    df_food_study_1.to_sql(table_name, db, chunksize=2000)
    db.close()

def helper_load_df_from_db(db_name, table_name) :

    db_file = PureWindowsPath(data_dir.joinpath(db_name + ".db"))
    db = sqlite3.connect(db_file.as_posix())

    df = pd.read_sql_query("select * from " + table_name, con=db)
    db.close()
    return df

def helper_store_csv_to_db(data_file, db_name, table_name) :
 
    db_file = PureWindowsPath(data_dir.joinpath(db_name + ".db" ))
    db = sqlite3.connect(db_file.as_posix())
    
    # lecture par bloc : chunksize correspond au nombre de lignes lues par bloc
    for chunk in pd.read_csv(data_file, chunksize=2000):
        chunk.to_sql(name=table_name, con=db, if_exists="append", index=False)  

#-------------------------------------------------------------------------
# sources des données brutes
#-------------------------------------------------------------------------

working_dir = PureWindowsPath(os.getcwd())
data_dir = PureWindowsPath(working_dir.joinpath('projects'))
data_result_name = 'result.csv'

# data_file = PureWindowsPath(data_dir.joinpath('en.openfoodfacts.org.products.tsv'))

df_food = helper_load_df_from_db("food", "data")

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
ser_column_null.sort_index(inplace=True)
ser_column_null.to_csv(data_dir.joinpath(data_result_name))

#-------------------------------------------------------------------------
# gérer les doublons 
# en l'occurence, pour les colonnes selectionnées, il n'y a pas de doublon
# Cepedant, si l'on réduit à nouveau les critères, il est probable que 
# des doublons apparaissent à nouveau (cf. plus loin)
#-------------------------------------------------------------------------
    
if df_food.drop_duplicates().shape == df_food.shape :
    print ("le tableau ne contient aucune ligne dupliquée")

#-------------------------------------------------------------------------
# La motivation de ce qui suit est d'extraire un sous-ensemble 
# de critères aussi large que possible en recherchant à réduire au 
# maximum les valeurs non nulles
#
# Différents "thresh" sont appliqués aux données
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
# le niveau de 150'000 définit une sélection de 36 critères (comprenant 
# tous les critères nécessaires à nos analyses)
#-------------------------------------------------------------------------
df_food_sel = df_food.dropna(thresh=150000, axis=1)
    
#-------------------------------------------------------------------------
# autorise l'affichage de toutes les colonnes
#-------------------------------------------------------------------------
pd.set_option('display.max_columns', df_food_sel.shape[1]) 
df_food_sel.head()

#-------------------------------------------------------------------------
# suppression de toute les colonnes qui contiennent les mots 
# completed
#-------------------------------------------------------------------------
    
# par exemple la colonne "states_tags" contient en grand nombre le texte "completed"
# qui n'est pas relevant pour notre étude

df_food_sel.states_tags.where(lambda x : x.str.contains("completed")).count()

# suppression de toutes les colonnes dont le nom contient "states" et "tags" 
# et qui contiennent "completed" en grand nombre

df_food_sel = df_food_sel.drop([col for col in df_food_sel.columns if "states" in col], axis=1)
df_food_sel = df_food_sel.drop([col for col in df_food_sel.columns if "tags" in col], axis=1)

#-------------------------------------------------------------------------
# ETUDE 1
# 
# repartition par macronutriment
# conserver également la liste des ingredients
# préparation de la base de données retenues
#
# Le sous-ensemble de colonnes ci-dessous est conservé
#-------------------------------------------------------------------------
column_to_keep = set(['product_name', 'countries','ingredients_text','nutrition_grade_fr','energy_100g',
                'fat_100g','proteins_100g', 'carbohydrates_100g'])

df_food_study_1 = df_food_sel.drop(axis=1, columns=set(df_food_sel.columns).difference(column_to_keep))

# indication d'un index et tri sur celui-ci
df_food_study_1.set_index(['product_name'], inplace=True)
df_food_study_1.sort_index(inplace=True)

#-------------------------------------------------------------------------
# cela est insuffisant : en effet, au niveau de l'index, il reste des doublons
#-------------------------------------------------------------------------    
# export de l'index pour se rendre compte 
df_food_study_1.index.to_series().to_csv(data_dir.joinpath(data_result_name))

#-------------------------------------------------------------------------    
# ceci permet d'extraire toutes les colonnes dont l'INDEX présente les doublons
#-------------------------------------------------------------------------    
df_food_study_1.loc[df_food_study_1.index.duplicated(),:]

# et inversément (sans doublons selon l'INDEX) à l'aide du ~ ..magique
df_food_study_1 = df_food_study_1.loc[~df_food_study_1.index.duplicated(),:]

#-------------------------------------------------------------------------
# gestion des doublons 
#-------------------------------------------------------------------------
df_food_study_1.drop_duplicates(inplace=True)

# helper_store_df_to_db(df_food_study_1, "df_food_study_1", "   df_food_study_1")