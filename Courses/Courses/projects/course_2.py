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
    # translate an ingredient (one at a time) in english with googletrans
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
# différents "thresh" (par step de 10'000) sont appliqués aux données du df
#------------------------------------------------------------------------------
def define_thresh_value (df_food_study) :

    df = df_food_study.copy()
    
    null_values_max = df.isnull().sum().max()
    if null_values_max > 0:
        threshes = np.arange(0, null_values_max, 10000)
     
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
    
        # construction d'un df à partir d'une liste de lignes construites
        # sur la base d'un dictionnaire
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
    # parmi toutes les colonnes numériques, supprimer les valeurs extrêmes

    for col_name in utilities.select_column_label(df_food_study, float):
        df_food_study = utilities.remove_outliers(df_food_study,col_name)
    
    helper_store_df_to_db(df_food_study, "df_food_study_1", "df_food_study_2")

#------------------------------------------------------------------------------
# ETUDE C : sub functions 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# return series of ingredients of randomly aliment pick from df_food_study
#------------------------------------------------------------------------------
def get_aliment_ingredients(df_food_study):

    aliment = df_food_study.sample(1)
        
    ingredients = aliment.ingredients_text.str.replace(r'[-|_|(|)|.|,|*|%|#|:|\'\]\[]','')
    ingredients = ingredients.str.split()
    ingredients = ingredients.get_values()[0]
    ser_ingredients = pd.Series(ingredients)

    return (aliment, ser_ingredients)

#------------------------------------------------------------------------------
# extrait un échantillon aléatoire parmi tous les aliments de df_food_study
#
# sur la base de cet échantillon, produit un df ("dictionnaire") 
# contenant les colonnes (ingredient_name, occurrences)
# 
# les ingrédients du dictionnaire sont triés par occurence décroissante et  
# limités au nombre de "ingredient_count"
#------------------------------------------------------------------------------
def build_aliment_ingredient_dictionnary(df_food_study, sample_size:int, ingredient_count:int):

    dict_ingredient = {}
    f_build_ingredient_dictionary = utilities.build_ingredient_dictionary(dict_ingredient)

    for i in np.arange(sample_size):
        ser_ingredients = get_aliment_ingredients(df_food_study)[1]
        f_build_ingredient_dictionary(ser_ingredients.to_json())

    df_dict = pd.DataFrame()
    df_dict = df_dict.from_dict(dict_ingredient, orient="index", columns=['occurences'])
    df_dict.sort_values('occurences', ascending=False, inplace=True)
    df_dict = df_dict.iloc[0:ingredient_count,]

    return df_dict

#------------------------------------------------------------------------------
# ETUDE C : main functions
#
# Analyse de la fréquence d'apparition des ingrédients dans les produits issus
# de df_food_study
#------------------------------------------------------------------------------
def analyze_ingredients_frequency():

    df_food_study = helper_load_df_from_db("df_food_study_1", "df_food_study_2")
    df_food_study.set_index(['product_name'], inplace=True)

    aliment_sample_size = 1000
    ingredient_sample_size = 100
 
    df_dict = build_aliment_ingredient_dictionnary(df_food_study, aliment_sample_size, ingredient_sample_size)
    df_dict = clean_ingredient_dictionnary(df_dict)
    df_dict.sort_values('occurences', ascending=False, inplace=True)
    
    df_dict = df_dict.iloc[0:30,] 
 
    fig, ax = plt.subplots()
    y_pos = np.arange(len(df_dict))
    ax.barh(y_pos, df_dict.occurences, align='center', color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_dict.index)
    ax.invert_yaxis() 
    ax.set_xlabel("Occurrences")
    ax.set_title("The 30 most common ingredients founded in " + str(aliment_sample_size) + " aliments")

    plt.show()

#------------------------------------------------------------------------------
# ETUDE B : sub functions
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# un aliment équilibré est selon la définition qui suit,
# celui qui dans sa décomposition en macro-nutriments
# est la plus proche de celle d'une alimentation équilibrée, soit une répartition
# de 65% en glucides, 25% en protéines, 10% en lipides
# retourne une copie 
#------------------------------------------------------------------------------
def compute_repartition_score(df_food_study, df_reference):

    df = df_food_study.copy()
    df = df.loc[:,col_macro_nutrients]
    df = df.sub(df_reference.loc['repartition'], axis=1)
    df = np.power(df_food_study_1, 2)
    
    df_food_study['repartition_score'] = np.power(df.sum(axis=1), 0.5)
    df_food_study.sort_values(by = 'repartition_score', inplace=True)
    df_food_study.sort_values(by = ['repartition_score','nutrition_grade_fr'], inplace=True)
        
    # ... plot 

    return df_food_study

#------------------------------------------------------------------------------
# contrôle que la valeur énergétique annoncées des aliments est conforme
# à celle recalculée en tenant compte des quantités et du pouvoir 
# calorique de chaque macro-nutriment
#
# Les produits dont les données caloriques semblent incohérentes sont supprimés
#
# retourne un df qui contient la quantité de joules par macro-nutriments
#------------------------------------------------------------------------------
def check_energy_tot(df_food_study, col_macro_nutrients, df_reference_value):

    col_delta_energy = 'delta_energy'

    col_for_check = col_macro_nutrients.copy()
    col_for_check.append('energy_100g')
        
    df = df_food_study.copy()
    df = df.loc[:, col_for_check]
    df_reference_value = df_reference_value.reindex(col_for_check, axis=1).fillna(1)

    # muliplication des quantités de chaque macro-nutriments par les calories 
    # qui leur correspond
    df = df.mul(df_reference_value.loc['energy_g'], axis=1)
    
    # convertir en joules 
    df.iloc[:, 0:3] = df.iloc[:, 0:3] * 4.18

    # calcul du delta relatif par rapport aux valeurs annoncées
    ser_energy_tot = df.loc[:, 'energy_100g']
    df[col_delta_energy] = (df.iloc[:, 0:3].sum(axis=1) - ser_energy_tot) / ser_energy_tot
        
    # ne conserver que les produits dont l'écart calorique entre la valeur annoncée et 
    # celle calculée diffère de moins 5%
    df = df[df[col_delta_energy].between(-0.05, 0.05)]
 
    return df

#------------------------------------------------------------------------------
# pour chaque aliment, caluler la proportion de calories apportées par chacun 
# des macro-nutriments
# retourne une copie 
#------------------------------------------------------------------------------
def compute_nutrient_breakdown_ratio(col_macro_nutrients, df_reference_value):
    
    df_food_study = helper_load_df_from_db("df_food_study_1", "df_food_study_2")
    df_food_study.set_index(['product_name'], inplace=True)
    df_food_study.sort_index(inplace=True)
 
    df = check_energy_tot(df_food_study, col_macro_nutrients, df_reference_value)
        
    df['ratio_cal_fat'] = df.fat_100g / df.energy_100g
    df['ratio_cal_carbohydrates'] = df.carbohydrates_100g / df.energy_100g
    df['ratio_cal_proteins'] = df.proteins_100g / df.energy_100g
     
    return df

#------------------------------------------------------------------------------
# ETUDE B : plot the result
#------------------------------------------------------------------------------
def plot_nutrient_breakdown_ratio(df_food_study, macro_nutrients):

    titles = pd.Series(macro_nutrients).apply(lambda x : str.upper(x)).tolist()
    col_ratios = pd.Series(macro_nutrients).apply(lambda x : "ratio_cal_" + x).tolist()
    labels = pd.Series(macro_nutrients).apply( lambda x : "Percentage of " + x)
    
    colors = ['red', 'blue', 'green']

    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=False)
    fig.suptitle('For each macronutrients, list of aliments with percentage in descending order' +
                 '(from a sample of size' + str(df_food_study.shape[0]))
  
    for ratio in col_ratios : 
        idx_ratio = col_ratios.index(ratio)

        df = df_food_study.copy()

        # on s'affranchit des valeurs > 1
        df = df[df.loc[:, ratio] <=1]
        df.sort_values(by=ratio, ascending=False, inplace=True)  
        df = df.loc[:, ratio]
        df = df.iloc[:10,]

        y_pos = np.arange(len(df))

        axes[idx_ratio].axes.barh(y_pos, df * 100, align='center', color=colors[i_ratio])
        axes[idx_ratio].axes.set_yticks(y_pos)
        axes[idx_ratio].axes.set_yticklabels(df.index.tolist(), minor=False)
        axes[idx_ratio].axes.invert_yaxis() 

        axes[idx_ratio].set_xlabel(labels[i_ratio])
        axes[idx_ratio].set_title(titles[i_ratio])
  
    fig.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
# ETUDE B : main function
#------------------------------------------------------------------------------
def analyze_nutrients_breakdown():

    macro_nutrients = ['fat', 'carbohydrates', 'proteins'] 
    col_macro_nutrients = pd.Series(macro_nutrients).apply(lambda x : x + "_100g").tolist()

    df_reference_value = pd.DataFrame(data = [(15,25,60), (9,4,4)], 
                                      columns = col_macro_nutrients,
                                      index = ['repartition', 'energy_g'])

    df_food_study = compute_nutrient_breakdown_ratio(col_macro_nutrients, df_reference_value)

    # restricts to a sample of size 1000 
    df_sample_food_study = df_food_study.sample(1000)
    
    plot_nutrient_breakdown_ratio(df_sample_food_study, macro_nutrients)

#------------------------------------------------------------------------------
# ETUDE E : sub functions
#
# mise en place d'une base de données normalisées; les ingrédients sont 
# extraits de la listes globales des aliments et stockés dans une table distinctes
#
# a table containing aliments and ingredients is built
# 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# translate all entry of the index in a column called 'ingredient_en' and set
# the index on this one
#------------------------------------------------------------------------------
def clean_ingredient_dictionnary(df_dict):

    words_to_exclude = pd.Series(['A','THE','FROM', 'AND', 'OR','AND/OR', 'DE', 'ET', 'IN', 'OF', 
                                  'TO','AT', '&', 'PASTEURIZED', 'ENRICHED', 'MODIFIED',
                                  'CONTAINS', 'MALTED', 'LESS'])

    f_translate = utilities.translate_ingredient()
    df_dict['ingredient_en'] = df_dict.index.to_series().apply(f_translate)

    df_dict.reset_index(inplace=True)
    df_dict.set_index('ingredient_en', inplace=True)
    df_dict.drop(words_to_exclude, errors="ignore", inplace=True)

    return df_dict    
        
#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def prepare_normalization():
    
    df_food_study = helper_load_df_from_db("df_food_study_1", "df_food_study_2")

    df_all_aliment = pd.DataFrame()
    df_all_aliment_ingredient = pd.DataFrame()

    df_global_dict = build_aliment_ingredient_dictionnary(df_food_study, 1000, 100)
    df_global_dict = clean_ingredient_dictionnary(df_global_dict)

    dict_aliment_ingredient = {}
    f_build_ingredient_dictionary = utilities.build_ingredient_dictionary(dict_aliment_ingredient)
    
    for i in np.arange(100):
        (df_aliment, ser_ingredients) = get_aliment_ingredients(df_food_study)
        f_build_ingredient_dictionary(ser_ingredients.to_json())
        
        df_aliment_ingredient = pd.DataFrame().from_dict(dict_aliment_ingredient, orient="index", columns=['occurences'])
    
        idx_ingredients_in_dict = df_aliment_ingredient.index.intersection(df_global_dict.index)
        idx_ingredients_in_dict = idx_ingredients_in_dict.drop_duplicates()

        df = pd.DataFrame(index=idx_ingredients_in_dict, columns=['product_name'])
        df = df.fillna(str(df_aliment.product_name.values.item(0)))
        df = df.reset_index()
        df.columns = ['ingredient_name', 'product_name']
         
        df_all_aliment_ingredient = pd.concat([df_all_aliment_ingredient, df])
        df_all_aliment = pd.concat([df_all_aliment, df_aliment])

        dict_aliment_ingredient.clear()

    return (df_all_aliment, df_all_aliment_ingredient)

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def build_normalized_database():

    (df_all_aliment, df_all_aliment_ingredient) = prepare_normalization()

    helper_store_df_to_db(df_all_aliment, 'aliment_1', 'aliment')
    helper_store_df_to_db(df_all_aliment_ingredient, 'aliment_1', 'aliment_ingredient')


#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def test_database():

    db_file = PureWindowsPath(data_dir.joinpath('aliment_1.db'))
    db = sqlite3.connect(db_file.as_posix())
   
    #-------------------------------------------------------------------------
    # load du df à partir de la base de données
    #-------------------------------------------------------------------------

    # tous les aliments qui contiennent du sucr
    df_q = pd.read_sql_query("select * from aliment as a, aliment_ingredient as ai where ai.product_name = a.product_name", con=db)
    df_q = pd.read_sql_query("select * from aliment_ingredient ai where ai.ingredient_name = 'SALT' ", con=db)