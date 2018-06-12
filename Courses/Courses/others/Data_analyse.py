import os
from pathlib import PureWindowsPath
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

#---- charger les données 

try :

    xmlFileName = 'O:\\Dupont\\24 BenAdmin-Param\\05 Paiement\\SEPA\\5_adresse_paiement_correction_is_iban.xlsx'
    
    currentDir = PureWindowsPath(os.getcwd())
    sampleDir = PureWindowsPath(currentDir.joinpath('WorldsTallestMountains'))
    dataFile = PureWindowsPath(sampleDir.joinpath('Mountains.csv'))

    data = pd.DataFrame()
    data = pd.read_csv(dataFile.as_posix())

    dataxml = pd.read_excel(xmlFileName)
    
    # clean up des données

    data.set_index('Mountain', inplace=True)
    data.drop(['Rank','Height (ft)','Coordinates', 'Parent mountain'], axis=1, inplace=True)
    data.drop(['Mount Everest / Sagarmatha / Chomolungma', 'Muztagh Ata'], axis=0, inplace=True)
        
    # supprimer toutes les lignes qui contiennent des valeurs non définies

    data.dropna(inplace=True)
    data = data[data['First ascent'] != 'unclimbed']

    # on ne peut pas changer de type sur des lignes qui contiennent des colonnes non définies 
    # ainsi le dropna ci-dessus

    # up casting 
    data['First ascent'] = data['First ascent'].astype(int)
    data['Ascents bef. 2004'] = data['Ascents bef. 2004'].astype(int)
    data['Failed attempts bef. 2004'] = data['Failed attempts bef. 2004'].astype(int)

    # création de nouvelles colonnes dans le dataframe

    data['Total attempts'] = data['Ascents bef. 2004'] + data['Failed attempts bef. 2004']
    data['Success rate'] = (data['Ascents bef. 2004'] / data['Total attempts'] )*100
    data['Difficulty'] = (data['Total attempts'] / data['Success rate'] )*100
    data['Difficulty'] = data['Difficulty'] / data['Difficulty'].max()
    data = data.sort_values(by = 'Difficulty', ascending = False)

    # affichage des données

    plt.scatter(data['Height (m)'], data['Total attempts'])
    plt.show()

except ValueError as e  :
    print ("ERROR")
    