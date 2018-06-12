
import os
from sqlalchemy.dialects.mssql.information_schema import columns
from pathlib import PureWindowsPath
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

try :

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
  
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(y)), x, align='center', color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y)
    ax.invert_yaxis()  
    ax.set_xlabel('Happiness Score')
    ax.set_title('Happiness Score of the 10 Happiest Countries in the World')
    plt.show()

except ValueError as e  :
    print (e)
       