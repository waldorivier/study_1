import os
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import random
import googletrans as translator
from pandas.tseries.offsets import *
import calendar

#-------------------------------------------------------------------------
# répertoire de travail
#-------------------------------------------------------------------------
working_dir = PureWindowsPath(os.getcwd())
data_dir = PureWindowsPath(working_dir.joinpath('projects'))
data_result_name = 'result.csv'

pd.set_option('display.max_columns', 30)

#-------------------------------------------------------------------------
# Class utilities
# - translation 
# - outliers
# - words dictionnary
#-------------------------------------------------------------------------
class utilities:
    #-------------------------------------------------------------------------
    # retourne la liste des noms de colonne d'un type 
    #-------------------------------------------------------------------------
    def select_columns(df, type):
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
    # translate a word (one at a time) in english with googletrans
    #-------------------------------------------------------------------------
    def translate_word():

        trans = translator.Translator()

        def f(x : str):
            try :
                if not x.isalpha():
                    return x

                elif x == "SEL":
                    return "SALT"

                else :
                    # t = trans.translate(x)
                    # return t.text

                    return x

            except ValueError:
                print ("erreur dans la traduction", x)
        
        return f

    #-------------------------------------------------------------------------
    # fill a dictionnary with all word and count occurences
    # dict_words : dictionnary filled
    #-------------------------------------------------------------------------
    def build_word_dictionary(dict_words):
        assert type(dict_words) is dict
   
        def f(x : str):
            try :
                ser_words = pd.read_json(x, typ="records")
                for item in ser_words.iteritems():
                    word = item[1].upper()

                    if (dict_words.get(word) is not None) :
                        dict_words[word] += 1
                    else :
                        dict_words[word] = 1
        
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

    df.to_sql(table_name, db, chunksize=2000, if_exists='replace')
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
    
    for chunk in pd.read_csv(data_file, chunksize=2000):
        chunk.to_sql(name=table_name, con=db, if_exists="append", index=False)  

    db.close()
 
#-------------------------------------------------------------------------
# ETUDE A : analyse meta-data and clean-up
# 
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# original file "en.openfoodfacts.org.products.tsv" est chargé dans une base 
# will be loaded and stored in a database name 'food.db' table name  'data' 
#-------------------------------------------------------------------------
def read_raw_data():

    data_file = PureWindowsPath(data_dir.joinpath('en.openfoodfacts.org.products.tsv'))
    helper_store_csv_to_db(data_file, "food", "data")

#------------------------------------------------------------------------------
# export to csv file all the columns containing null values
#------------------------------------------------------------------------------
def export_null_columns(df, data_file) : 
   
    ser_column_null = df.isnull().sum()
    ser_column_null.sort_index(inplace=True)
    ser_column_null.to_csv(data_file)

#------------------------------------------------------------------------------
# analyze columns 
#
# applies series of threshes (by step 10000) on the df and mesures the number
# of remaining columns
# 
# plot the result  
#------------------------------------------------------------------------------
def analyze_columns(df_food_study) :

    df = df_food_study.copy()
    
    null_values_max = df.isnull().sum().max()
    if null_values_max > 0:
        threshes = np.arange(0, null_values_max, 10000)
     
        rows = []

        def select_columns(df, thresh):
            df = df.dropna(thresh=thresh, axis=1)
        
            dict = {'_thresh' : 0, '_shape' : 0}
            dict['_thresh'] = thresh
            dict['_shape'] = df.shape[1]
    
            rows.append(dict)
        
        for thresh in threshes :
            select_columns (df, thresh)
    
        # build a dataframe based on dict of rows
        df_result = pd.DataFrame(rows)  
        df_result.set_index('_shape', inplace=True)
    
        df_result.to_csv(data_dir.joinpath(data_result_name))
 
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.suptitle("numbers of remaining columns in terms of thresh for a df of shape (" +  
                        str(df_food_study.shape[0]) + "," + str(df_food_study.shape[1]) + ")")
        
        x_pos = np.arange(len(df_result))
        ax.bar(x_pos, df_result._shape, align='center', color='green')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_result._thresh, rotation=45)

        ax.set_ylabel("remaining columns")
        ax.set_xlabel("thresh")

        plt.show()

#------------------------------------------------------------------------------
# First step of data cleaning
#
# Columns will be reduced to a level that minimize the lost of relevant 
# data 
#
# resulting df will be store in database 
#------------------------------------------------------------------------------
def clean_raw_data():

    df_food = helper_load_df_from_db("food", "data")

    # eliminates all columns with all null values
    # df will be reduced from 163 columns to 147
    df_food.dropna(axis=1, how="all", inplace=True)

    # check for duplicated 
    if df_food.drop_duplicates().shape == df_food.shape:
        print ("df doesn't contain duplicated data")
        
    analyze_columns(df_food)
   
    # le niveau de 150'000 définit une sélection de 36 critères (comprenant 
    # tous les critères nécessaires à nos analyses)
    df_food_cl = df_food.dropna(thresh=150000, axis=1)
    
    # autorise l'affichage de toutes les colonnes
    pd.set_option('display.max_columns', df_food_cl.shape[1]) 
    df_food_cl.head()

    # eliminates columns containing word 'completed' which seems not to be relevant
    # example : counting  occurrences  for column 'tags' 
    df_food_cl.states_tags.where(lambda x : x.str.contains("completed")).count()

    # eliminates columns whose name contains 'states' or 'tags
    df_food_cl = df_food_cl.drop([col for col in df_food_cl.columns if "states" in col], axis=1)
    df_food_cl = df_food_cl.drop([col for col in df_food_cl.columns if "tags" in col], axis=1)

    helper_store_df_to_db(df_food_cl, "df_food_cl", "df_food_cl")

#------------------------------------------------------------------------------
# Second step of data cleaning
#
# The dataset is based on that previously cleaned
# 
# Mainly eliminates all null values, duplicated  and numerical outliers
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

    # eliminates duplicates
    df_food_study.drop_duplicates(inplace=True)

    # gérer les valeurs extrêmes
    # parmi toutes les colonnes numériques, supprimer les valeurs extrêmes

    for col_name in utilities.select_columns(df_food_study, float):
        df_food_study = utilities.remove_outliers(df_food_study, col_name)
    
    # stores the resulting cleaned datas to database
    helper_store_df_to_db(df_food_study, "df_food_study_1", "df_food_study_2")

#-----------------------------------------------------------------------------
# ETUDE B : nutrient repartition
# sub functions
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# A balanced aliment is defined as follow :
#
# aliments which repartition approximate a balanced alimentation that is:
# 65% glucides, 25% proteins, 10% fats
#
# compute the distance between each aliment repartition and reference's aliment
# repartition define in df_reference_value
# 
# return df with added repartition_score colum
#------------------------------------------------------------------------------
def compute_repartition_score(df_food_study, col_macro_nutrients, df_reference_value):

    col_repartition_score = 'repartition_score'

    df = df_food_study.copy()
    df = df.loc[:,col_macro_nutrients]

    # calculate distance
    df = df.sub(df_reference_value.loc['repartition'], axis=1)
    df = np.power(df, 2)
    df[col_repartition_score] = np.power(df.sum(axis=1), 0.5)

    df = df_food_study.merge(df, left_index=True, right_index=True)

    # sort  
    df.sort_values(by = [col_repartition_score], inplace=True)
        
    return df

#------------------------------------------------------------------------------
# checks that energetic value announced in the data is comform to that which 
# is re-calculated 
#
# aliments whose energetic value seems inconsistent will be removed 
#
# return df containing calories per macronutrients
#------------------------------------------------------------------------------
def check_energy_tot(df_food_study, col_macro_nutrients, df_reference_value):

    col_delta_energy = 'delta_energy'

    col_for_check = col_macro_nutrients.copy()
    col_for_check.append('energy_100g')
        
    df = df_food_study.copy()
    df = df.loc[:, col_for_check]
    df_reference_value = df_reference_value.reindex(col_for_check, axis=1).fillna(1)

    # multiply macronutrients quantities with correponding calories / gramm
    df = df.mul(df_reference_value.loc['energy_g'], axis=1)
    
    # conversion to joules
    df.iloc[:, 0:3] = df.iloc[:, 0:3] * 4.18

    # calculate delta energy compared to annouced value
    ser_energy_tot = df.loc[:, 'energy_100g']
    df[col_delta_energy] = (df.iloc[:, 0:3].sum(axis=1) - ser_energy_tot) / ser_energy_tot
        
    # retains only aliments which delta lies below 5% (in absolute value)
    df = df[df[col_delta_energy].between(-0.05, 0.05)]
 
    return df

#------------------------------------------------------------------------------
# for each aliment, compute the percentage of calories provided from 
# each macronutrients
# 
# return a df with ratio added columns
#------------------------------------------------------------------------------
def compute_nutrient_breakdown_ratio(df_food_study, col_macro_nutrients, df_reference_value):
    
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
    labels = pd.Series(macro_nutrients).apply( lambda x : "Calories percentage of " + x)
    
    colors = ['red', 'blue', 'green']

    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=False)
    fig.suptitle("Proportion of calories from macronutrients " 
                 + "(10 aliments from a sample of size " + str(df_food_study.shape[0]) + ")")
  
    for ratio in col_ratios : 
        i_ratio = col_ratios.index(ratio)

        df = df_food_study.copy()

        # the (few) values greater than 1 will be removed
        df = df[df.loc[:, ratio] <=1]
        df.sort_values(by=ratio, ascending=False, inplace=True)  
        df = df.loc[:, ratio]
        df = df.iloc[:10,]

        y_pos = np.arange(len(df))

        axes[i_ratio].axes.barh(y_pos, df * 100, align='center', color=colors[i_ratio])
        axes[i_ratio].axes.set_yticks(y_pos)

        # limits product name to 30 lexical characters
        axes[i_ratio].axes.set_yticklabels(df.index.str[0:30].tolist(), minor=False)
        axes[i_ratio].axes.invert_yaxis() 

        axes[i_ratio].set_xlabel(labels[i_ratio])
        axes[i_ratio].set_title(titles[i_ratio])
  
    fig.tight_layout()
    plt.show()

def plot_nutrient_repartition_score(df_food_study):

    col_repartition_score = 'repartition_score'

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle("The " + str(df_food_study.shape[0]) + " more balanced aliments according definition")
  
    df = df_food_study.loc[:,col_repartition_score]
    df = df.iloc[:30,]

    y_pos = np.arange(len(df))
    ax.barh(y_pos, df, align='center', color='green')
    ax.set_yticks(y_pos)

    # limits product name to 30 lexical characters
    ax.set_yticklabels(df.index.str[0:30].tolist())
    ax.invert_yaxis() 

    ax.set_xlabel("Score (distance to aliment's reference repartition)")
    
    fig.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
# ETUDE B : nutrient repartition
#
# main function
#------------------------------------------------------------------------------
def analyze_data():

    macro_nutrients = ['fat', 'carbohydrates', 'proteins'] 
    col_macro_nutrients = pd.Series(macro_nutrients).apply(lambda x : x + "_100g").tolist()

    df_reference_value = pd.DataFrame(data = [(15,60,25), (9,4,4)], 
                                      columns = col_macro_nutrients,
                                      index = ['repartition', 'energy_g'])

    df_food_study = helper_load_df_from_db("df_food_study_1", "df_food_study_2")
    df_food_study.set_index(['product_name'], inplace=True)
    df_food_study.sort_index(inplace=True)

    # 1 . nutrient breakdown
    df = compute_nutrient_breakdown_ratio(df_food_study, col_macro_nutrients, df_reference_value)

    # restricts to a sample of size 1000 
    df = df.sample(1000)
    
    plot_nutrient_breakdown_ratio(df, macro_nutrients)

    # 2 . repartition score
    df = compute_repartition_score(df_food_study, col_macro_nutrients, df_reference_value)

    # restricts to a sample of size 30 
    df = df.iloc[0:30,]

    plot_nutrient_repartition_score(df)

#------------------------------------------------------------------------------
# ETUDE C :  Analyze ingredient's frequency
# 
# sub functions 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# return a series of ingredients of randomly aliment retrieve from df_food_study
#------------------------------------------------------------------------------
def get_aliment_ingredients(df_food_study):

    aliment = df_food_study.sample(1)
        
    ingredients = aliment.ingredients_text.str.replace(r'[-|_|(|)|.|,|*|%|#|:|\'\]\[]','')
    ingredients = ingredients.str.split()
    ingredients = ingredients.get_values()[0]
    ser_ingredients = pd.Series(ingredients)

    return (aliment, ser_ingredients)

#------------------------------------------------------------------------------
# retrieve a sample of aliments from df_food_study 
# based on this sample, a dictionnary of ingredients with columns (ingredient_name, occurrences),
# will be build 
#
# în this dictionnary, ingredients will be sorted by 'occurences' and the shape will be limited to
# ingredient_count
#
# return a df containing a dictionnary of ingredients
#------------------------------------------------------------------------------
def build_aliment_ingredient_dictionnary(df_food_study, sample_size:int, ingredient_count:int):

    dict_ingredients = {}
    f_build_ingredient_dictionary = utilities.build_word_dictionary(dict_ingredients)

    for i in np.arange(sample_size):
        ser_ingredients = get_aliment_ingredients(df_food_study)[1]
        f_build_ingredient_dictionary(ser_ingredients.to_json())

    df_dict = pd.DataFrame()
    df_dict = df_dict.from_dict(dict_ingredients, orient="index", columns=['occurences'])
    df_dict.sort_values('occurences', ascending=False, inplace=True)
    df_dict = df_dict.iloc[0:ingredient_count,]

    return df_dict

#------------------------------------------------------------------------------
# ETUDE C : main functions
#
# Analyze ingredient's frequency of ingredients used in aliments of df_food_study
#------------------------------------------------------------------------------
def analyze_ingredients_frequency():

    df_food_study = helper_load_df_from_db("df_food_study_1", "df_food_study_2")
    df_food_study.set_index(['product_name'], inplace=True)

    aliment_sample_size = 1000
    ingredient_sample_size = 100
 
    df_dict = build_aliment_ingredient_dictionnary(df_food_study, 
                                                   aliment_sample_size, 
                                                   ingredient_sample_size)

    df_dict = clean_word_dictionnary(df_dict)
    df_dict.sort_values('occurences', ascending=False, inplace=True)
    
    df_dict = df_dict.iloc[0:30,] 
 
    fig, ax = plt.subplots()
    y_pos = np.arange(len(df_dict))
    ax.barh(y_pos, df_dict.occurences, align='center', color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_dict.index)
    ax.invert_yaxis() 
    ax.set_xlabel("Occurrences")
    ax.set_title("The 30 most common ingredients founded in " + 
                 str(aliment_sample_size) + ") aliments")

    plt.show()

#------------------------------------------------------------------------------
# ETUDE D : TIMESERIES 
# 
# sub fonctions
# 
#------------------------------------------------------------------------------
def compute_mean_elapsed(df_food_study):

    ser_create = pd.to_datetime(df_food_study.created_datetime)
    ser_modify = pd.to_datetime(df_food_study.last_modified_datetime)

    # verify that no null values in both series
    if(ser_create.isnull().sum() == 0 and ser_modify.isnull().sum() == 0):

        # compute difference
        ser_delta = ser_modify - ser_create

        # verify that no negative values in the difference
        if(ser_delta.where(lambda x : x < pd.Timedelta(0)).count() == 0):
            mean_delta = ser_delta.mean()

    return mean_delta

#------------------------------------------------------------------------------
# plot created entries per years / month
#------------------------------------------------------------------------------
def plot_years_month_entries(df_food_study):

    month = calendar.month_name[1:]

    ser_create = pd.to_datetime(df_food_study.created_datetime)
    df = pd.DataFrame(ser_create, columns = ['created_datetime', 'year', 'month'])

    df.set_index('created_datetime', inplace=True)
    df.year = df.index.year
    df.month = df.index.month

    df.reset_index(inplace=True)
    df.set_index(['year', 'month'], inplace=True)

    df_g = np.log(df.groupby(['year', 'month']).count())
    
    # define plot parameters
    ax = df_g.unstack(level=0).plot.bar(width=0.8)
    x_pos = np.arange(len(month))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(month, rotation=45)
    ax.set_ylabel("number of created aliment's entries")

    ax.set_xlabel("Months")
    ax.set_title("(logarithm of) number of created aliment's entries in the database")
    ax.legend(labels=df_g.index.levels[0])

    plt.show()

#------------------------------------------------------------------------------
# plot mean created entries per month over the timeline
#------------------------------------------------------------------------------
def plot_mean_month_entries(df_food_study):

    month = calendar.month_name[1:]

    ser_create = pd.to_datetime(df_food_study.created_datetime)
    df = pd.DataFrame(ser_create, columns = ['created_datetime', 'year', 'month'])

    df.set_index('created_datetime', inplace=True)
    df.year = df.index.year
    df.month = df.index.month

    df.reset_index(inplace=True)
    df.set_index(['year', 'month'], inplace=True)

    df_g = np.log(df.groupby(['month','year']).count())
    mean = df_g.groupby('month').mean()
    std = df_g.groupby('month').std()
    
    # define plot parameters
    fig, ax = plt.subplots()
    x_pos = np.arange(len(month))
    ax.bar(x_pos, mean['created_datetime'], align='center', 
           color='blue', yerr=std['created_datetime'], ecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(month, rotation=45)
    ax.set_ylabel("mean of created aliment's entries")

    ax.set_xlabel("Months")
    ax.set_title("(logarithm of) mean of created aliment's entries in the database")
  
    plt.show()
        
#------------------------------------------------------------------------------
# ETUDE D : main function
#------------------------------------------------------------------------------
def analyze_time_series():

    df_food_study = helper_load_df_from_db("df_food_study_1", "df_food_study_2")

    print(compute_mean_elapsed(df_food_study))

    plot_mean_month_entries(df_food_study)
    plot_years_month_entries(df_food_study)

#------------------------------------------------------------------------------
# ETUDE E : DATABASE 
# 
# sub functions
# 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# translate all entry of the df_dict index in a column called 'ingredient_en' and set
# the index on this one
#
# return a df with added columns 'ingredient_en'
#------------------------------------------------------------------------------
def clean_word_dictionnary(df_dict):

    words_to_exclude = pd.Series(['A','THE','FROM', 'AND', 'OR','AND/OR', 'DE', 'ET', 'IN', 'OF', 
                                  'TO','AT', '&', 'PASTEURIZED', 'ENRICHED', 'MODIFIED',
                                  'CONTAINS', 'MALTED', 'LESS'])

    f_translate = utilities.translate_word()
    df_dict['ingredient_en'] = df_dict.index.to_series().apply(f_translate)

    df_dict.reset_index(inplace=True)
    df_dict.set_index('ingredient_en', inplace=True)
    df_dict.drop(words_to_exclude, errors="ignore", inplace=True)

    return df_dict    
        
#------------------------------------------------------------------------------
# prepare normalized data 
#
# build a dictionnary of ingredients
# for 1000 aliments randomly selected from df_food_study_2, extract its ingredients 
# and join them with those in dictionnary
#
# one table with aliment and one table with only product_name and ingredient_name
# will be built
#------------------------------------------------------------------------------
def prepare_normalization():
    
    df_food_study = helper_load_df_from_db("df_food_study_1", "df_food_study_2")

    df_all_aliment = pd.DataFrame()
    df_all_aliment_ingredient = pd.DataFrame()

    df_global_dict = build_aliment_ingredient_dictionnary(df_food_study, 1000, 100)
    df_global_dict = clean_word_dictionnary(df_global_dict)

    dict_aliment_ingredient = {}
    f_build_ingredient_dictionary = utilities.build_ingredient_dictionary(dict_aliment_ingredient)
    
    for i in np.arange(1000):
        (df_aliment, ser_ingredients) = get_aliment_ingredients(df_food_study)
        f_build_ingredient_dictionary(ser_ingredients.to_json())
        
        df_aliment_ingredient = pd.DataFrame().from_dict(dict_aliment_ingredient, orient="index", columns=['occurences'])
    
        i_ingredients_in_dict = df_aliment_ingredient.index.intersection(df_global_dict.index)
        i_ingredients_in_dict = i_ingredients_in_dict.drop_duplicates()

        df = pd.DataFrame(index=i_ingredients_in_dict, columns=['product_name'])
        df = df.fillna(str(df_aliment.product_name.values.item(0)))
        df = df.reset_index()
        df.columns = ['ingredient_name', 'product_name']
         
        df_all_aliment_ingredient = pd.concat([df_all_aliment_ingredient, df])
        df_all_aliment = pd.concat([df_all_aliment, df_aliment])

        dict_aliment_ingredient.clear()

    return (df_all_aliment, df_all_aliment_ingredient)

#------------------------------------------------------------------------------
# save normalized df to database df_food.db
#------------------------------------------------------------------------------
def build_normalized_database():

    (df_all_aliment, df_all_aliment_ingredient) = prepare_normalization()

    helper_store_df_to_db(df_all_aliment, 'df_food', 'aliment')
    helper_store_df_to_db(df_all_aliment_ingredient, 'df_food', 'aliment_ingredient')

    # set a primary key on table aliment
    
    db_file = PureWindowsPath(data_dir.joinpath('df_food.db'))
    db = sqlite3.connect(db_file.as_posix())
    c  = db.cursor()

    # SQLLITE doesn't support ALTER ADD / DROP 
    # c.execute("ALTER TABLE ALIMENT ADD PRIMARY KEY (product_name)")

    # retrieve ddl, modify it and replay

#------------------------------------------------------------------------------
# test database queries on df_food.db
#------------------------------------------------------------------------------
def perform_database_queries():

    # build_normalized_database()

    db_file = PureWindowsPath(data_dir.joinpath('df_food.db'))
    db = sqlite3.connect(db_file.as_posix())

    # query all aliments which contains SALT
    df_q = pd.read_sql_query("select distinct(a.product_name) from aliment as a, aliment_ingredient as ai where ai.product_name = " +
                             " a.product_name and ai.ingredient_name = 'SALT'", con=db)

    return df_q
    
#------------------------------------------------------------------------------
# ETUDE F : CORRELATION 
# 
# sub fonctions
#------------------------------------------------------------------------------
def analyze_correlations(): 

    score_col = 'nutrition-score-fr_100g'
    cols_to_drop = set(['code', 'additives_n', 'ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n'])
    cols_to_drop.add(score_col)
    
    df_food_study = helper_load_df_from_db("df_food_study_1", "df_food_study_2")

    # retains only float columns
    num_cols = utilities.select_columns(df_food_study, float)
    
    # eliminates all colums define above
    predictors_cols = set(num_cols).difference(cols_to_drop)

    # sample of size 1000 is significant enough
    df = df_food_study.sample(1000)
   
    df_corr = df.corr()
    df_corr = df_corr.sort_values(by=score_col, ascending = False)
    
    sns.pairplot(df, 
                 x_vars=predictors_cols,
                 y_vars=[score_col], 
                 kind="reg",
                 plot_kws={'line_kws':{'color':'red'}})
    plt.show()

    return df_corr[score_col]

#------------------------------------------------------------------------------
# ETUDE F : CORRELATION 
# 
# sub fonctions
#------------------------------------------------------------------------------
def check_nutrition_score_grade_relation():

    df_food_study = helper_load_df_from_db("df_food_study_1", "df_food_study_2")

    df = df_food_study.loc[:,['nutrition_grade_fr', 'nutrition-score-fr_100g']]
    
    grade_convertion  = {'a':'1', 'b' : '2', 'c' : '3','d' : '4', 'e' : '5'}
    df.nutrition_grade_fr = df.nutrition_grade_fr.apply(lambda x : grade_convertion.get(x))
    df = df.astype(float)    
    sns.pairplot(df, 
                 x_vars=['nutrition_grade_fr'],
                 y_vars=['nutrition-score-fr_100g'], 
                 kind="reg",
                 plot_kws={'line_kws':{'color':'red'}})
    plt.show()
