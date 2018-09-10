import os
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sqlite3
import random
import googletrans as translator

#-------------------------------------------------------------------------
# répertoire de travail
#-------------------------------------------------------------------------
working_dir = PureWindowsPath(os.getcwd())
data_dir = PureWindowsPath(working_dir.joinpath('projects'))
data_result_name = 'result.csv'

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
class utilities:
    #-------------------------------------------------------------------------
    # retourne la liste des noms de colonne d'un type donné
    #-------------------------------------------------------------------------
    def select_column_label(df, type):
        columns = df.dtypes[df.dtypes.apply(lambda x : x == type)].index.tolist()
        return columns

    #-------------------------------------------------------------------------
    # identifier et gérer les valeurs extrêmes 
    # on considère le seuil de 1% pour déterminer les valeurs extrêmes; ce seuil
    # determine (sous l'hypothèse de normalité de la distribution) toutes les valeurs 
    # x | x > abs(mean - 3 * std) 
    #-------------------------------------------------------------------------
    def remove_outliers(df, col_name):

        def reject(ser : pd.Series):
            mean = ser.mean() 
            std  = 3 * ser.std()
  
            def f(x):
                reject = False

                if np.abs(x - mean) > std : 
                    reject = True
                return reject
    
            return f

        ser_col = df.loc[:,col_name]
        f_reject = reject(ser_col)
        df = df.loc[~ser_col.apply(f_reject)]

        return df

    #-------------------------------------------------------------------------
    # traduction au moyen de googletrans
    #-------------------------------------------------------------------------
    def translate_ingredient():

        trans = translator.Translator()

        def f(x : str):
            try :
                if not x.isalpha():
                    return x

                elif x == "SEL":
                    return "SALT"

                else :
                    t = trans.translate(x)
                    return t.text

            except ValueError:
                print ("erreur dans la traduction", x)
                
        
        return f

    #-------------------------------------------------------------------------
    # alimente un dictionnaire de tous les ingrédients rencontrés
    # et en compte les occurences
    #-------------------------------------------------------------------------
    def build_ingredient_dictionary(dict_ingredients):
        assert type(dict_ingredients) is dict
   
        def f(x : str):
            try :
                ser_ingredients = pd.read_json(x, typ="records")
                for item in ser_ingredients.iteritems():
                    ingredient = item[1].upper()

                    if (dict_ingredients.get(ingredient) is not None) :
                        dict_ingredients[ingredient] += 1
                    else :
                        dict_ingredients[ingredient] = 1
        
            except Exception:   
                print ("build dictionary error")

        return f 

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

    db.close()
 
#-------------------------------------------------------------------------
# le fichier contenant les données brutes 
# "en.openfoodfacts.org.products.tsv" est chargé dans une base 
# de données food.db qui comporte une seule table "data"
#-------------------------------------------------------------------------
def read_raw_data():

    data_file = PureWindowsPath(data_dir.joinpath('en.openfoodfacts.org.products.tsv'))
    helper_store_csv_to_db(data_file, "food", "data")

#------------------------------------------------------------------------------
# exporter les colonnes d'un df dans un fichier csv avec le nombre 
# de valeurs nulles afin de visualiser les caractéristiques qui ne sont 
# pas relevantes / ou qui ne nous interesent tout simplement pas 

#------------------------------------------------------------------------------
def export_null_columns(df, data_file) : 
   
    ser_column_null = df.isnull().sum()
    ser_column_null.sort_index(inplace=True)
    ser_column_null.to_csv(data_file)

#------------------------------------------------------------------------------
# la motivation de la fonction ci-dessous est de définir un sous-ensemble 
# de critères aussi large que possible en recherchant à réduire au 
# maximum les valeurs non nulles
#
# Différents "thresh" sont appliqués aux données du df; on fixe un step de 10'000
#------------------------------------------------------------------------------
def define_thresh_value (df) :

    null_values_max = df.isnull().sum().max()
    if null_values_max > 0:
        threshes = np.arange(0, null_values_max, 10000)

        # construction d'un df à partir d'une liste de lignes construites
        # sur la base d'un dictionnaire
        rows = []

        def optimize_col_selection(thresh):
            df = df.dropna(thresh=thresh, axis=1)
        
            dict = {'thresh_' : 0, 'schape_':0, 'mean_':0}
            dict['thresh_'] = thresh
            dict['schape_'] = df.shape[1]
            dict['mean_']   = df.isnull().sum().mean()

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

#------------------------------------------------------------------------------
# première étape de nettoyage des données
# produit un df qui contient un premier ensemble de données épuréees
# qui sont sauvegardées en base de données 
#------------------------------------------------------------------------------
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

    # suppression de toutes les colonnes qui contiennent les mots "completed"
    # par exemple la colonne "states_tags" contient en grand nombre le texte "completed"
    # qui n'est pas relevant pour notre étude
    df_food_cl.states_tags.where(lambda x : x.str.contains("completed")).count()

    # suppression de toutes les colonnes dont le nom contient "states" et "tags" 
    # et qui contiennent "completed" en grand nombre
    df_food_cl = df_food_cl.drop([col for col in df_food_cl.columns if "states" in col], axis=1)
    df_food_cl = df_food_cl.drop([col for col in df_food_cl.columns if "tags" in col], axis=1)

    # sauvegarde en base de données
    helper_store_df_to_db(df_food_cl, "df_food_cl", "df_food_cl")

#------------------------------------------------------------------------------
# ETUDE 1.0
# 
# préparation de la base de données après une seconde passe d'épuration
#------------------------------------------------------------------------------
def setup_db_study():

    df_food_study = helper_load_df_from_db("df_food_cl", "df_food_cl")
                  
    # suppression de toutes les valeurs nulles
    df_food_study.dropna(inplace=True)

    # indication d'un index et tri sur celui-ci
    df_food_study.set_index(['product_name'], inplace=True)
    df_food_study.sort_index(inplace=True)
    df_food_study.sort_values(by='last_modified_datetime', ascending = False, inplace=True)

    # export de l'index pour se rendre compte des doublons
    df_food_study.index.to_series().to_csv(data_dir.joinpath(data_result_name))

    # ceci permet d'extraire toutes les colonnes dont l'index présente les doublons
    df_food_study.loc[df_food_study.index.duplicated(),:]

    # et inversément (sans doublons selon l'index) à l'aide du ~...magique
    df_food_study = df_food_study.loc[~df_food_study.index.duplicated(),:]

    # gestion des doublons 
    df_food_study.drop_duplicates(inplace=True)

    # gérer les valeurs extrêmes
    # parmi toutes les colonnes numériques, supprimer les valeurs extrêmes correspondantes

    for col_name in utilities.select_column_label(df_food_study, float):
        df_food_study = utilities.remove_outliers(df_food_study,col_name)
    
    helper_store_df_to_db(df_food_study, "df_food_study_1", "df_food_study_2")

#------------------------------------------------------------------------------
# ETUDE C
# 
# Analyse de la fréquence d'apparition d'ingrédients dans les produits issus
# de df_food_study
#------------------------------------------------------------------------------
def analyze_ingredients_frequency():

    # après relecture depuis la base de données, il faut rétablr l'index
    df_food_study = helper_load_df_from_db("df_food_study_1", "df_food_study_2")

    df_food_study.set_index(['product_name'], inplace=True)
    df_food_study.sort_index(inplace=True)

    #------------------------------------------------------------------------------

    df_food_study.ingredients_text = df_food_study.ingredients_text.str.split()

    #------------------------------------------------------------------------------
    # execute un extraction d'un échantillon aléatoire parmi tous les produits
    # de df_food_study
    #
    # retourne un df ("dictionnaire") contenant les colonnes (ingredient_name, occurrences); 
    # les ingrédients du dictionnaire sont triés par occurence et ils sont 
    # limités au nombre de "ingredient_count"
    #------------------------------------------------------------------------------
    def product_ingredient_sample(sample_size:int, population_size:int, ingredient_count:int):

        dict_ingredients = {}
        f_build_ingredient_dictionary = utilities.build_ingredient_dictionary(dict_ingredients)

        i = 0
        while i < sample_size :
            try :
                ingredients = df_food_study.iloc[random.randrange(population_size), 
                                                 df_food_study.columns.get_loc('ingredients_text')]

                ser_ingredients = pd.Series(ingredients)
                ser_ingredients = ser_ingredients.str.replace(r'[-|_|(|)|.|,|*|%|#|:|\'\]\[]','')
                ser_ingredients = ser_ingredients[~ser_ingredients.apply(lambda x : x  == '')]
 
                f_build_ingredient_dictionary(ser_ingredients.to_json())

            except ValueError :
                print(i)
                print (ser_ingredients)
  
            i += 1
            if i > sample_size :
                break

        # le dictionnaire de l'échantillon est trié par occurence décroissante
        # est la taille limitée à ingredient_count

        df_sample = pd.DataFrame()
        df_sample = df_sample.from_dict(dict_ingredients, orient="index", columns=['occurences'])
        df_sample.sort_values('occurences', ascending=False, inplace=True)
        df_sample = df_sample.iloc[0:ingredient_count,]

        return df_sample

    #------------------------------------------------------------------------------
    # ANALYSE de la fréquence des ingredients evaluées à partir de plusieurs échantillons 
    # de produits
    #
    # les "dictionnaires" produit par les échatillons sont consolidés 
    #------------------------------------------------------------------------------

    df_dicts = pd.DataFrame()

    # les mots qui ne correspondent pas à des ingrédients et que l'on veut par conséquent exclure
    word_to_exclude = pd.Series(['FROM', 'AND', 'DE', 'ET', 'IN', 'OF', 'AT'])

    sample_size = 100
    ingredient_count = 100
 
    nb_sample = 0
    while nb_sample < 5 : 
        df_dict = product_ingredient_sample(sample_size, df_food_study.shape[0], ingredient_count)
  
        f_translate = utilities.translate_ingredient()
        df_dict['ingredient_en'] = df_dict.index.to_series().apply(f_translate)

        df_dict.reset_index(inplace=True)
        df_dict.set_index('ingredient_en', inplace=True)

        df_gr_ingredients = df_dict.groupby('ingredient_en').sum()
        df_gr_ingredients.sort_values('occurences', ascending=False, inplace=True)
        df_gr_ingredients = df_gr_ingredients.drop(word_to_exclude, errors="ignore")

        df_dicts = pd.concat([df_dicts, df_gr_ingredients.transpose()], sort=False)

        nb_sample += 1

    df_dicts = df_dicts.transpose()
    df_dicts.fillna(0, inplace=True)
    df_dicts['occurences_mean'] = df_dicts.mean(axis=1)
    df_dicts.sort_values('occurences_mean', ascending=False, inplace=True)

    # on ne retient que les 10 ingrédients les plus féquents

    df_ingredient_l = df_dicts.iloc[0:30,] 
 
    fig, ax = plt.subplots()
    y_pos = np.arange(len(df_ingredient_l))
    ax.barh(y_pos, df_ingredient_l.occurences_mean, align='center', color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_ingredient_l.index)
    ax.invert_yaxis() 
    ax.set_xlabel('Nombre d''occurences pour 100 produits')
    ax.set_title('Nombre d''occurences des ingrédients')

    plt.show()


analyze_ingredients_frequency()



if 0:
    #------------------------------------------------------------------------------
    # ETUDE B
    #------------------------------------------------------------------------------
  
    macro_nutrients = ['fat_100g','proteins_100g','carbohydrates_100g']

    df_reference = pd.DataFrame(data = [(15,25,60), (9,4,4)], 
                                columns = macro_nutrients,
                                index = ['repartition', 'energy_g'])

    #------------------------------------------------------------------------------
    # un aliment équilibré est selon la définition qui suit,
    # celui qui dans sa décomposition en macro-nutriments
    # est la plus proche de celle d'une alimentation équilibrée, soit une répartition
    # de 65% en glucides, 25% en protéines, 10% en lipides
    # retourne une copie 
    #------------------------------------------------------------------------------
    def compute_repartition_score(df_food_study, df_reference):

        df = df_food_study.copy()
        df = df.loc[:,macro_nutrients]
        df = df.sub(df_reference.loc['repartition'], axis=1)
        df = np.power(df_food_study_1, 2)
    
        df_food_study['repartition_score'] = np.power(df.sum(axis=1), 0.5)
        df_food_study.sort_values(by = 'repartition_score', inplace=True)
        df_food_study.sort_values(by = ['repartition_score','nutrition_grade_fr'], inplace=True)
        
        # ... plot 

        return df_food_study

    #------------------------------------------------------------------------------
    # pour chaque aliment, caluler la proportion de calories apportées par chacun 
    # des macro-nutriments
    # retourne une copie 
    #------------------------------------------------------------------------------
    def compute_nutrient_breakdown_ratio(df_food_study, df_reference):
       
        columns = macro_nutrients.copy()
        columns.append('energy_100g')
        
        df = df_food_study.copy()
        df = df.loc[:, columns]
        df_reference = df_reference.reindex(columns, axis=1).fillna(1)

        df = df.mul(df_reference.loc['energy_g'], axis=1)
        
        # plausibiler la valeur énergétique annoncée du produit 
        energy_tot = df.iloc[:,3]
        ser_plausibility = (df.iloc[:,0:3].sum(axis=1).mul(4.18) - energy_tot) / energy_tot
        
        # identifier et supprimer les valeurs infinies
        inf_or_nan = ser_plausibility.isin([np.inf, -np.inf, np.nan])
        ser_plausibility = ser_plausibility[~inf_or_nan]

        # identifier et supprimer les valeurs extrêmes

        ser_plausibility = utilities.remove_outliers(pd.DataFrame(ser_plausibility, columns=['variation']), 
                                                     'variation').variation

        return df[~inf_or_nan]
    
    df_food_study = helper_load_df_from_db("df_food_study_1", "df_food_study_2")
    df_food_study.set_index(['product_name'], inplace=True)
    df_food_study.sort_index(inplace=True)

    compute_repartition_score(df_food_study, df_refercence)
          
#------------------------------------------------------------------------------
# ETUDE E
#
# mise en place d'une base de données normalisées 
#------------------------------------------------------------------------------