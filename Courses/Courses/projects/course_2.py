import os
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sqlite3
import utilities as util

#-------------------------------------------------------------------------
# répertoire de travail
#-------------------------------------------------------------------------
working_dir = PureWindowsPath(os.getcwd())
data_dir = PureWindowsPath(working_dir.joinpath('projects'))
data_result_name = 'result.csv'

#-------------------------------------------------------------------------
# persiste / charge un DataFrame vers / de la  base de données 
# REM : db_name sans extension ; le nom de la table correspond à celui de 
#       la base de données
#-------------------------------------------------------------------------
def helper_store_df_to_db(df, db_name, table_name) :

    db_file = PureWindowsPath(data_dir.joinpath(db_name + ".db" ))
    db = sqlite3.connect(db_file.as_posix())

    df.to_sql(table_name, db, chunksize=2000)
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
    
    # lecture par bloc : chunksize correspond au nombre de lignes lues puis inserées en base
    # de données

    for chunk in pd.read_csv(data_file, chunksize=2000):
        chunk.to_sql(name=table_name, con=db, if_exists="append", index=False)  

#-------------------------------------------------------------------------
# le fichier contenant les données brutes 
# "en.openfoodfacts.org.products.tsv" est chargé dans une base 
# de données food.db qui comporte une seule table "data"
#-------------------------------------------------------------------------
def read_raw_data():

    data_file = PureWindowsPath(data_dir.joinpath('en.openfoodfacts.org.products.tsv'))
    helper_store_csv_to_db(data_file, "food", "data")

#-------------------------------------------------------------------------
# exporter les colonnes d'un df dans un fichier csv avec le nombre 
# de valeurs nulles afin de visualiser les caractéristiques qui ne sont 
# pas relevantes / ou qui ne nous interesent tout simplement pas 
#-------------------------------------------------------------------------
def export_null_columns(df, data_file) : 
   
    ser_column_null = df.isnull().sum()
    ser_column_null.sort_index(inplace=True)
    ser_column_null.to_csv(data_file)

#-------------------------------------------------------------------------
# La motivation de l'opération ci-dessous est de définir un sous-ensemble 
# de critères aussi large que possible en recherchant à réduire au 
# maximum les valeurs non nulles
#
# Différents "thresh" sont appliqués aux données du df 
#-------------------------------------------------------------------------
def define_thresh_value (df) :

    threshes = np.arange(0, df.isnull().sum().max(), 10000)

    # construction d'un df à partir d'une liste de lignes construites
    # sur la base d'un dictionnaire
    rows = []

    def optimize_col_selection(thresh):
        df_ = df.dropna(thresh=thresh, axis=1)
        
        dict = {'thresh_' : 0, 'schape_':0, 'mean_':0}
        dict['thresh_'] = thresh
        dict['schape_'] = df_.shape[1]
        dict['mean_']   = df_.isnull().sum().mean()

        rows.append(dict)
        
    for thresh in threshes :
        optimize_col_selection (thresh)
    
    df_result = pd.DataFrame(rows)  
    df_result.set_index('schape_', inplace=True)
    
    # le tableau ci-dessous indique le nombre de critères conservés et 
    # la moyenne des valeurs nulles des colonnes résiduelles conservées
    # par niveau (thresh) en fonction duquel une colonne est supprimée.   
    df_result.to_csv(data_dir.joinpath(data_result_name))

    df_result.plot(figsize=(16,12))
    plt.xlabel('nombre de colonnes')
    plt.legend(labels=['moyenne du nombre de valeur nulle','niveau'])

    plt.show()

#-------------------------------------------------------------------------
# première étape de nettoyage des données
# produit un df qui contient un premier ensemble de données épuréees
# qui sont sauvegardées en base de données 
#-------------------------------------------------------------------------
def clean_raw_data():

    df_food = helper_load_df_from_db("food", "data")

    # nettoyage des données
    # suppression de toutes les colonnes qui n'ont pas de données
    # réduction du tableau à 147 colonnes (initialement 163)
    df_food.dropna(axis=1, how="all", inplace=True)

    # gérer les doublons 
    # en l'occurence, pour les colonnes selectionnées, il n'y a pas de doublon
    # Cepedant, si l'on réduit à nouveau les critères, il est probable que 
    # des doublons apparaissent à nouveau (cf. plus loin)
    if df_food.drop_duplicates().shape == df_food.shape :
        print ("le tableau ne contient aucune ligne dupliquée")
        
    define_thresh_value(df_food)

    # le niveau de 150'000 définit une sélection de 36 critères (comprenant 
    # tous les critères nécessaires à nos analyses)
    df_food_cl = df_food.dropna(thresh=150000, axis=1)
    
    # autorise l'affichage de toutes les colonnes
    pd.set_option('display.max_columns', df_food_cl.shape[1]) 
    df_food_cl.head()

    # suppression de toute les colonnes qui contiennent les mots "completed"
    # par exemple la colonne "states_tags" contient en grand nombre le texte "completed"
    # qui n'est pas relevant pour notre étude
    df_food_cl.states_tags.where(lambda x : x.str.contains("completed")).count()

    # suppression de toutes les colonnes dont le nom contient "states" et "tags" 
    # et qui contiennent "completed" en grand nombre
    df_food_cl = df_food_cl.drop([col for col in df_food_cl.columns if "states" in col], axis=1)
    df_food_cl = df_food_cl.drop([col for col in df_food_cl.columns if "tags" in col], axis=1)

    # sauvegarde en base de données
    helper_store_df_to_db(df_food_cl, "df_food_cl", "df_food_cl")

#-------------------------------------------------------------------------
# ETUDE 1
# 
# repartition par macronutriment
# conserver également la liste des ingredients
# préparation de la base de données
#-------------------------------------------------------------------------
def setup_db_study_1():

    df_food_study_1 = helper_load_df_from_db("df_food_cl", "df_food_cl")

    # le sous-ensemble comprenant les colonnes ci-dessous est conservé
    column_to_keep = set(['product_name', 'created_datetime', 'last_modified_datetime', 
                          'countries','ingredients_text','nutrition_grade_fr','energy_100g',
                          'fat_100g','proteins_100g', 'carbohydrates_100g'])

    df_food_study_1 = df_food_study_1.drop(axis=1, 
                                     columns=set(df_food_study_1.columns).difference(column_to_keep))

    # suppression de toutes les valeurs NAN
    df_food_study_1.dropna(inplace=True)

    # indication d'un index et tri sur celui-ci
    df_food_study_1.set_index(['product_name'], inplace=True)
    df_food_study_1.sort_index(inplace=True)

    # export de l'index pour se rendre compte des doublons
    df_food_study_1.index.to_series().to_csv(data_dir.joinpath(data_result_name))

    # ceci permet d'extraire toutes les colonnes dont l'INDEX présente les doublons
    df_food_study_1.loc[df_food_study_1.index.duplicated(),:]

    # et inversément (sans doublons selon l'INDEX) à l'aide du ~ ..magique
    df_food_study_1 = df_food_study_1.loc[~df_food_study_1.index.duplicated(),:]

    # gestion des doublons 
    df_food_study_1.drop_duplicates(inplace=True)

    # gérer les valeurs extrêmes
    # parmi toutes les colonnes numériques, supprimer les valeurs extrêmes correspondantes
    for col_name in util.utilities.select_column_label(df_food_study_1, float):
        df_food_study_1 = util.utilities.remove_outliers(df_food_study_1,col_name)
    
        # print (df_food_study_1.shape) 

    helper_store_df_to_db(df_food_study_1, "df_food_study_1", "df_food_study_1")


df_food_study_1 = helper_load_df_from_db("df_food_study_1", "df_food_study_1")
df_food_study_1.set_index(['product_name'], inplace=True)
df_food_study_1.sort_index(inplace=True)

# parsing ingredients

def translate_ingredient() :
    translator = Translator()

    def f_(x : str):
        if not x.isalpha():
            return x
        else :
            t = translator.translate(x)
            return t.text

    return f_

# traitement d'un produit

f_translate_ingredient = translate_ingredient()

df_food_study_1.ingredients_text = df_food_study_1.ingredients_text.str.split()

ser_ingredients = pd.Series(df_food_study_1.ingredients_text[0])
ser_ingredients = ser_ingredients.str.replace(r'[_|(|)|.|,]','')
ser_ingredients = ser_ingredients[~ser_ingredients.apply(lambda x : x == ':')]
ser_ingredients.apply(f_translate_ingredient)


