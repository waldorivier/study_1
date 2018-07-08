#
# lecture du fichier DTA 
# extraction des destinataires et de leur montant associé

import os
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
import xml.etree.cElementTree as et

try :

    dta_file_name = '_5750_18_101_qfzj46001.xml'
    comptes_techniques_file_name = 'comptes_techniques.xlsx'
    data_result_merge_file_name = 'merge.csv'
    data_result_dta_file_name = 'dta.csv'
    data_result_comptes_file_name = 'comptes.csv'

    #--------------------------------------------------------------------------
    # chemin d'accès aux fichiers
    #--------------------------------------------------------------------------
    
    local = True

    if local :
        working_dir = PureWindowsPath(os.getcwd())
        data_dir = PureWindowsPath(working_dir.joinpath('Data'))
    
    else :
        client_root_dir = PureWindowsPath('O:\\')
        data_dir = PureWindowsPath(client_root_dir.joinpath('Lausanne','24 BenAdmin-Param','05 Paiement','SEPA','Contrôles'))
    
    #--------------------------------------------------------------------------
    # lecture du fichier DTA
    #--------------------------------------------------------------------------
    
    data_dta = PureWindowsPath(data_dir.joinpath(dta_file_name))
    data_result_comptes = PureWindowsPath(data_dir.joinpath(data_result_comptes_file_name))
    data_result_dta = PureWindowsPath(data_dir.joinpath(data_result_dta_file_name))
    data_result_merge = PureWindowsPath(data_dir.joinpath(data_result_merge_file_name))
    dta_doc = et.parse(PureWindowsPath(data_dir.joinpath(dta_file_name).as_posix()))
    comptes_xl = PureWindowsPath(data_dir.joinpath(comptes_techniques_file_name))

    # liste pour rassembler les infos des destinataires (nom, montant)
    destinataires = []

    # sélection et extraction des éléments de paiement

    dta_doc_root = dta_doc.getroot()
    for node in dta_doc_root.findall('.//CdtTrfTxInf') :
        destinataire = node.find('.//Nm').text
        montant = node.find('.//InstdAmt').text
        destinataires.append ([destinataire, montant])
        
    df_cols = ['destinataire', 'Montant']
    df_dta = pd.DataFrame(destinataires, columns = df_cols)

    # conversion des montant en float (par défaut ils sont de type objet)

    df_dta.Montant = df_dta.Montant.astype(float)
    df_dta.Montant.sum()
    
    #--------------------------------------------------------------------------
    # export en csv / pour comparaison avec l'extraction des comptes ci-dessous
    #--------------------------------------------------------------------------

    # df_dta.to_csv(PureWindowsPath(data_dir.joinpath(data_result_file_name)), sep = ';')

    #--------------------------------------------------------------------------
    # lecture du fichier des comptes techniques pour comparaison
    #--------------------------------------------------------------------------

    df_comptes_orig = pd.read_excel(comptes_xl.as_posix())
  
    # ne conserver que les colonnes qui nous interessent
    
    df_comptes = df_comptes_orig.loc[:,['destinataire','RPAYX','IMSOU']]
   
    # définir une colonne pour le total
    df_comptes['TOTAL'] = df_comptes.RPAYX + df_comptes.IMSOU
    
    # Uppercase de tous les noms

    df_comptes.destinataire = df_comptes.destinataire.str.upper()
    df_dta.destinataire = df_dta.destinataire.str.upper()

    # somme par destinataire, le dataframe est réduit pour ne représenter 
    # plus que le regroupement

    df_comptes = df_comptes.groupby(by = 'destinataire', as_index=False)['TOTAL'].sum()
    df_comptes.to_csv(data_result_comptes, sep = ';')
   
    # procéder de même pour le dta
    df_dta = df_dta.groupby(by = 'destinataire', as_index=False)['Montant'].sum()
    df_dta.to_csv(data_result_dta, sep = ';')

    df_m = pd.merge(df_comptes, df_dta, how = "outer")
    df_m.to_csv(data_result_merge, sep = ';')
   
except ValueError as e  :
    print (e)



       
