#
# lecture du fichier DTA 
# extraction des destinataires et de leur montant associé

import os
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
import xml.etree.cElementTree as et

try :
    data_file_name = '_5750_18_101_qfzj46001.xml'
    data_result_file_name = 'dta.csv'

    # -- chemin d'accès au fichier 

    client_root_dir = PureWindowsPath('O:\\')
    working_dir = PureWindowsPath(os.getcwd())
    data_dir = PureWindowsPath(client_root_dir.joinpath('Lausanne','24 BenAdmin-Param','05 Paiement','SEPA','Contrôles'))
    data_file = PureWindowsPath(data_dir.joinpath(data_file_name))
    doc = et.parse(data_file.as_posix())

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

    df_dta.Montant.astype(float)
    df_dta.Montant.astype(float).sum()
    
    # export en csv 
    df_dta.to_csv(PureWindowsPath(data_dir.joinpath(data_result_file_name)), sep = ';')

except ValueError as e  :
    print (e)



       
