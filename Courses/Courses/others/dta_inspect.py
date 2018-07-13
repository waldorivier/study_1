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
    element_paiement_file_name = 'element_paiement.xlsx'
    type_comptes_techniques_file_name = 'type_comptes_techniques.xlsx'
    type_ecritures_file_name = 'type_ecritures.xlsx'

    result_merge_file_name = 'merge.csv'
    result_dta_file_name = 'dta.csv'
    result_comptes_file_name = 'comptes.csv'
    result_file_name = 'RESULT.csv'

    #--------------------------------------------------------------------------
    # chemin d'accès aux fichiers
    #--------------------------------------------------------------------------
    
    local = False

    if local :
        working_dir = PureWindowsPath(os.getcwd())
        data_dir = PureWindowsPath(working_dir.joinpath('Data'))
    
    else :
        client_root_dir = PureWindowsPath('O:\\')
        data_dir = PureWindowsPath(client_root_dir.joinpath('Lausanne','24 BenAdmin-Param','05 Paiement','SEPA','Contrôles'))
    
    #--------------------------------------------------------------------------
    def helper_get_file(file_name) :
        return PureWindowsPath(data_dir.joinpath(file_name).as_posix())

    #--------------------------------------------------------------------------
    # analyse des éléments de paiement 
    #--------------------------------------------------------------------------

    if 1 : 
        df_elem_paie = pd.read_excel(helper_get_file(element_paiement_file_name))
        
        # split compte / ecriture car ELPA contient la table associative (compte, ecriture)
        # conversion en numérique et conversion des valeurs nan afin de permettre la jointure 
        # pas permis de faire des jointures sur des colonnes contenant nan
        
        df_elem_paie['COMPTE'] = df_elem_paie.COMPTE_ELPA.str.slice(0,4).fillna(0).astype(int)
        df_elem_paie['ECRITURE'] = df_elem_paie.COMPTE_ELPA.str.slice(5,7)

        df_type_compte_technique = pd.read_excel(helper_get_file(type_comptes_techniques_file_name))
        df_type_ecriture = pd.read_excel(helper_get_file(type_ecritures_file_name))

        # conversion des objets en string pour permettre d'effectuer la jointure
        df_type_ecriture['TYPE_ECRIT'] = df_type_ecriture['TYPE_ECRIT'].astype(str)

        df_m = pd.merge(df_elem_paie, df_type_compte_technique, left_on=['COMPTE'], right_on=['TYPE_COMPTE'], how="left")
        df_m = pd.merge(df_m, df_type_ecriture, left_on=['ECRITURE'], right_on=['TYPE_ECRIT'], how="left")
    
        # colonnes à conserver 
        
        col_keep = set(['NO_CAS','NO_CATE','NOM_ELEM_PMT','GENRE_ELPA',
                        'NOM_ELEM_BAS','COMPTE','LIBF_TYCO','TYPE_ECRIT',
                        'LIBF_TYEC'])

        # on supprime toutes les colonnes autres que celles à conserver; sauvegarde du résultat
        df_m.drop(axis=1, columns=set(df_m.columns).difference(col_keep), inplace=True)
        df_m.to_csv(helper_get_file(result_merge_file_name), sep = ';')

        df_elem_paie.to_csv(helper_get_file(result_file_name), sep = ';')
    
    #--------------------------------------------------------------------------
      
    if 0 :
        dta_doc = et.parse(helper_get_file(dta_file_name))
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
        # lecture du fichier des comptes techniques pour comparaison
        #--------------------------------------------------------------------------

        df_comptes_orig = pd.read_excel(helper_get_file(comptes_techniques_file_name))
  
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
        df_comptes.to_csv(helper_get_file(result_compte_file_name), sep = ';')
   
        # procéder de même pour le dta
        df_dta = df_dta.groupby(by = 'destinataire', as_index=False)['Montant'].sum()
        df_dta.to_csv(helper_get_file(result_dta_file_name), sep = ';')

        df_m = pd.merge(df_comptes, df_dta, how = "outer")
        df_m.to_csv(helper_get_file(result_merge_file_name), sep = ';')

except ValueError as e  :
    print (e)



       
