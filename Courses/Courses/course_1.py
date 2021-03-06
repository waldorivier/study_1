# -----------------------------------------------------------------------------
# WALDO RIVIER
# 17.06.2018 
# Exercices du module 1 Applied Data Science: Machine Learning
# -----------------------------------------------------------------------------
import os

from pathlib import PureWindowsPath
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

try :

    #--------------------------------------------------------------------------
    # décomposition du score de bonheur par critère; forme stack bar 
    # fonction utilisée plus bas
    #--------------------------------------------------------------------------
    def plot_stack_bar (df, title) :

        x = df['Country']
        x_pos = np.arange(len(x))

        y = df['Happiness Score']
       
        fig, a2 = plt.subplots()
        a2.set_xticks(x_pos)
        a2.set_xticklabels(x, rotation=90)

        for i in range(3, 10) :
            ci = df.iloc[:,i]
            if i == 3 :
                a2.bar(x_pos, ci, color = 'silver')
                c = ci
            else :
                a2.bar(x_pos, ci, bottom = c)
                c = c + ci

        a2.legend(labels=df_hap_10.columns[3:10], loc="upper right", bbox_to_anchor=(1.2, 1))
        a2.set_xlabel('Country')
        a2.set_ylabel('Happiness Score')
        a2.set_title(title)
        plt.show()
  
   #-------------------------------------------------------------------------
   # lecture du fichier contenant les données du World 
    working_dir = PureWindowsPath(os.getcwd())
    data_dir = PureWindowsPath(working_dir.joinpath('Data'))
    data_file = PureWindowsPath(data_dir.joinpath('data.csv'))
    df = pd.read_csv(data_file.as_posix())

    df.info()
    df.columns
    df.dtypes

    df.count()
    df.dropna(inplace=True)
    df.count()
        
    # selection des colonnes de type numérique don on produit une copie
    df_num_col = df.select_dtypes(include=['float64', 'int64']).copy()

    # produit une série somme / max / min des valeurs prises par colonne
    sum_num_col = df_num_col.sum()

    print('-----------------------------------')
    print('Max value from numerical columns')
    print(df_num_col.max())   
    print('-----------------------------------')
    
    print('-----------------------------------')
    print('Min value from numerical columns')
    print(df_num_col.min())   
    
    df_hap_asc = df.sort_values(by = 'Happiness Rank', ascending = True)
    print('-----------------------------------')
    print('les 10 pays les + "heureux"')
    print(df_hap_asc.head(10))
    print('-----------------------------------')

    print('les 10 pays les - "heureux"')
    df_hap_desc = df.sort_values(by = 'Happiness Rank', ascending = False)
    print(df_hap_desc.head(10))
    print('-----------------------------------')

    # suppression de la région Europe compte tenu que des sous-régions
    # sont définies
        
    df_region = df.copy()
    df_region.set_index(['Region'], inplace=True)
    df_region.drop(['Europe'], axis=0, inplace=True)

    # une variable qui définit le groupe by
    ser_region_group = df_region.groupby(['Region'])['Happiness Score']
    
    print('moyenne des scores de bonheur par région')
    ser_region_hap = ser_region_group.mean()
    print(ser_region_hap.sort_values(ascending=False))
    print('-----------------------------------')
    
    print('nombre de pays, par région, qui font un score > 6')
    score_above_6 = df_region['Happiness Score'] > 6
    df_region_f = df_region[score_above_6]
    ser_nb_region_f = df_region_f.groupby(['Region'])['Happiness Score']
    print(ser_nb_region_f.count())
    print('-----------------------------------')
   
    print('distance, par région, entre le score le max et le score min')
    ser_dist_score = ser_region_group.max() - ser_region_group.min()
    ser_dist_score.sort_values(ascending=False)
    print(ser_dist_score)
    print('-----------------------------------')

    print('région avec la distance maximum entre le score le max et le score min')
    print(ser_dist_score.head(1))
    
    print('--------------------------------------')
    print('GRAPHIQUE les 10 + bar plot horizontal')
    print('--------------------------------------')
    
    df_hap_10 = df_hap_asc.head(10)
    y = df_hap_10['Country']
    y_pos = np.arange(len(y))
    x = df_hap_10['Happiness Score']

  
    fig, a1 = plt.subplots()
    a1.barh(np.arange(len(y)), x, align='center', color='green', ecolor='black')
    a1.set_yticks(y_pos)
    a1.set_yticklabels(y)
    a1.invert_yaxis()  
    a1.set_xlabel('Happiness Score')
    a1.set_title('Happiness Score of the 10 Happiest Countries in the World')
    plt.show()
    
    print('--------------------------------------')
    print('GRAPHIQUE les 10 + bar plot vertical  ')
    print('Décomposition par critère             ')
    print('--------------------------------------')
    
    plot_stack_bar (df_hap_10, 'Happiness Score of the 10 Happiest Countries in the World / Deccomposition')
    
    #--------------------------------------------------------------------------
    # Sélectionner tous les pays de la région 'Afrique'

    region = 'Africa'
    f_region = df['Region']==region
    df_r = df[f_region]
    df_r.sort_values(by = 'Happiness Rank', ascending = True)
    plot_stack_bar (df_r, 'Happiness Score of the All Countries in ' + region + ' / Decomposition')
  
    #--------------------------------------------------------------------------
    # Histogramme par JOB Satisfaction 
    #--------------------------------------------------------------------------
    
    sns.distplot(df['Job Satisfaction'], bins=6, kde=False, norm_hist=True)
    plt.show()

    #--------------------------------------------------------------------------
    # Pairwise Scatter PLOT
    #--------------------------------------------------------------------------
    
    # des variables "numérique", on retire "Happiness Score";
    # on construit alors les comparaisons entre chacune de ces variables 
    # et "Happiness Score"  
   
    comparison_vars = df_num_col.drop(['Happiness Score'], axis=1).columns.values
    g = sns.PairGrid(df, x_vars=comparison_vars, y_vars=["Happiness Score"])
    g = g.map(plt.scatter)
    plt.show()
     
    #--------------------------------------------------------------------------
    # Correlation
    #--------------------------------------------------------------------------

    hap_score_corr_values = df.corr()['Happiness Score'].sort_values(ascending=False)
    hap_score_corr_values.drop(['Happiness Score'], inplace = True)

    print('Meilleure correlation avec "Happiness Score" : ')
    print(hap_score_corr_values.head(1))

    #--------------------------------------------------------------------------
    # Probabilité
    #--------------------------------------------------------------------------

    df_west_europe_above_6 = df[df['Region']=='Western Europe']
    df_west_europe_above_6 = df_west_europe_above_6[df_west_europe_above_6['Happiness Score'] > 6]
        
    df_world_above_6 = df[df['Happiness Score'] > 6]
    prob = df_west_europe_above_6.Country.count() / df_world_above_6.Country.count()
    print('Probabilité que parmi les pays avec un "Happiness Score" > 6, celui-ci soit de l''Europe de l''ouest')
    print (prob)

    #-------------------------------------------------------------------------
    # Matrice
    #--------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # function lambda pour convertir en 0 ou 1 les éléments d'une série si 
    # l'élément est égal / différent de c
    #-------------------------------------------------------------------------
    def f_region(r) :
        def f(x) :
            b = 0
            if x == r :
                b = 1
            else : 
                b = 0
            return b
        return f

    #-------------------------------------------------------------------------
        df_countries = df.copy()
        df_countries = df.set_index(['Country'])
        ser_country_region = df_countries['Region']
    
    for r in ser_country_region.unique() :
        # on ajoute une colonne pour chaque région dans laquelle on associe les régions
       
        df_countries[r] = ser_country_region.values
        # pour chacune des colonnes, on masque les régions qui ne correspondent pas
        # au "titre" de cette colonne (i.e colonne 'Africa'-> toutes les régions de
        # on fait appel a une fonction type lambda
        df_countries[r] = df_countries[r].apply(f_region(r))
        
    df_countries.as_matrix(columns=ser_country_region.unique())
    print (df_countries)

except ValueError as e  :
    print (e)
       