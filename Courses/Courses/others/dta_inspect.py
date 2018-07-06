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
    data_result_file_name = 'dta.csv'

    #--------------------------------------------------------------------------
    # chemin d'accès aux fichiers
    #--------------------------------------------------------------------------

    client_root_dir = PureWindowsPath('O:\\')
    working_dir = PureWindowsPath(os.getcwd())
    data_dir = PureWindowsPath(client_root_dir.joinpath('Lausanne','24 BenAdmin-Param','05 Paiement','SEPA','Contrôles'))
    
    #--------------------------------------------------------------------------
    # lecture du fichier DTA
    #--------------------------------------------------------------------------
    
    doc = et.parse(PureWindowsPath(data_dir.joinpath(dta_file_name).as_posix()))

    # liste pour rassembler les infos des destinataires (nom, montant)
    destinataires = []

    # sélection et extraction des éléments de paiement

    doc_root = doc.getroot()
    for node in doc_root.findall('.//CdtTrfTxInf') :
        destinataire = node.find('.//Nm').text
        montant = node.find('.//InstdAmt').text
        destinataires.append ([destinataire, montant])
        
    df_cols = ['destinataire', 'Montant']
    df_dta = pd.DataFrame(destinataires, columns = df_cols)

    # conversion des montant en float (par défaut ce sont des objet)

    df_dta.Montant = df_dta.Montant.astype(float)
    df_dta.Montant.sum()
    
    #--------------------------------------------------------------------------
    # export en csv / pour comparaison avec l'extraction des comptes ci-dessous
    #--------------------------------------------------------------------------

    df_dta.to_csv(PureWindowsPath(data_dir.joinpath(data_result_file_name)), sep = ';')

    #--------------------------------------------------------------------------
    # lecture du fichier des comptes techniques pour comparaison
    #--------------------------------------------------------------------------

    df_comptes_orig = pd.read_excel(PureWindowsPath(data_dir.joinpath(comptes_techniques_file_name)).as_posix(), sep = ';', encoding = "ascii")
  
    # ne conserver que les colonnes qui nous interessent
    
    df_comptes = df_comptes_orig.loc[:,['destinataire','RPAYX','IMSOU']]
   
    # définir une colonne pour le total
    df_comptes['TOTAL'] = df_comptes.RPAYX + df_comptes.IMSOU
    
    # somme par destinataire
    ser_comptes_dest = df_comptes.groupby(by = 'destinataire')['TOTAL']

    # contrôle de la somme totale
    ser_comptes_dest.sum().sum()
    
    # procéder de même pour le dta
    ser_dta_dest = df_dta.groupby(by = 'destinataire')['Montant']

    df_m = pd.merge(ser_comptes_dest, ser_df_dta, how = "inner")

except ValueError as e  :
    print (e)



       
