#
# lecture du fichier DTA 
# extraction des destinataires et de leur montant associé

import os
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
import xml.etree.cElementTree as et
from datetime import date
import datetime

pd.set_option('display.max_columns', 30)

try :
    dta_file_name = '_5750_18_101_qfzj46001.xml'
    comptes_techniques_file_name = 'comptes_techniques.xlsx'
    element_paiement_file_name = 'element_paiement.xlsx'
    type_comptes_techniques_file_name = 'type_comptes_techniques.xlsx'
    type_ecritures_file_name = 'type_ecritures.xlsx'
    liste_45_file_name = 'liste_45.xlsx'

    result_merge_file_name = 'merge.csv'
    result_dta_file_name = 'dta.csv'
    result_comptes_file_name = 'comptes.csv'
    result_file_name = 'RESULT.csv'

    pd.set_option('display.max_columns', 30)

    #--------------------------------------------------------------------------
    # chemin d'accès aux fichiers
    #--------------------------------------------------------------------------
    
    local = False

    if local :
        working_dir = PureWindowsPath(os.getcwd())
        data_dir = PureWindowsPath(working_dir.joinpath('Data'))

    else :
        client_root_dir = PureWindowsPath('O:\\')
        # data_dir = PureWindowsPath(client_root_dir.joinpath('UBS Optio 1e','24 BenAdmin-Param','06 Reprise','Atos','data_reprise'))
        data_dir = PureWindowsPath(client_root_dir.joinpath('lausanne','24 BenAdmin-Param','06 Reprise'))
        # data_dir = PureWindowsPath(client_root_dir.joinpath('cap','24 BenAdmin-Param','09 Listes collectives','extractions', '2019', '10_effectif_sig'))

    #--------------------------------------------------------------------------
    def helper_get_file(file_name) :
        return PureWindowsPath(data_dir.joinpath(file_name).as_posix())

    #--------------------------------------------------------------------------
    # analyse des éléments de paiement 
    #--------------------------------------------------------------------------

    if 0 : 
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

        #--------------------------------------------------------------------------
        # lecture des comptes / écriture de la liste 45
        #--------------------------------------------------------------------------

    if 0 : 
        df_liste_45 = pd.read_excel(helper_get_file(liste_45_file_name))
       
        # forcer en string pour l'évaluation de la regexp (un nombre unique étant considéré 
        # comme un... numérique)

        df_ecrit = df_liste_45.LTYPE_ECRIT.astype("str")
        df_ecrit = df_ecrit.str.extractall('(\d+)')
        df_ecrit.rename(columns={0:'ecrit'}, inplace = True)

        # nous sommes en présence d'un multi index de niveau 2
        df_ecrit.index.set_names(['type_compte','nb match'], inplace= True)

        # renommer l'index (indice) avec le no de type de compte 
        df_ecrit.index.set_levels(levels=[df_liste_45.TYPE_COMPTE.tolist(), np.arange(6)], inplace=True)
        df_ecrit.sort_values(by = 'ecrit')
        df_ecrit.to_csv(helper_get_file(result_file_name), sep = ';')
        

    if 0:
        df_data_converted = pd.read_excel(helper_get_file('chem_invalides.xlsx'), sheet_name='data_2')
        
        df_header = df_data_converted.iloc[0:3,:]
        df_data = df_data_converted.iloc[3:,:]
        # df_data = df_data_converted.iloc[4:5,:]
        
        # convert datetime to compatible mupe format
        date_columns = ['DAFFIL','DENSER','DATEFF','DANAIS']

        for column in date_columns:
            if column in df_data.columns:
                df_data[column] = df_data[column].map(lambda x : x.strftime('%d.%m.%Y')) 

        int_columns = ['NPOST']
        for column in int_columns:
            if column in df_data.columns:
                df_data[column] = df_data[column].astype(int)

        # df_data.fillna(method='ffill', inplace = True)

        # tentative de conversion en entier
        try:
            df_data['CODMOU'] = df_data['CODMOU'].astype(int)
        except:
            None

        df_header.to_csv(helper_get_file('header_chem.csv'), sep = ';', index=False)
        df_data.to_csv(helper_get_file('data_chem.csv'), sep = ';', header=False, index=False)

        df_data_extraction = pd.read_excel(helper_get_file('chem_invalides.xlsx'), sheet_name='EXTR')
        df_data_extraction.to_csv(helper_get_file('header_extraction.csv'), sep = ';', index=False)


    if 0:
        df_orig = pd.read_csv(helper_get_file('data_extr_actifs_raoj4045u.csv'), sep = ';')
        # df_orig = pd.read_csv(helper_get_file('data_ctrl_salaires_gar_prec_rakm3529e.csv'), sep = ';')
 
        df = df_orig.iloc[2:,:].copy()
        df = df[df.DTMUTX == '01.01.2019']
        df['NO_CAS'] = df['NO_CAS'].astype(int)
        
        df[df.NO_CAS == 1].RVGARX.astype(float).sum()
        df[df.NO_CAS == 2].RVGARX.astype(float).sum()

        df[df.NO_CAS == 1].SACGAR.astype(float).sum()
        df[df.NO_CAS == 2].SACGAR.astype(float).sum()
 
        df[df.NO_CAS == 1].RVGARX.astype(float).sum()

    if 0:
    
        df_gmes = pd.read_excel('C:\\Users\\A0710996\\Desktop\\SWISSSTAFF_SSE_repartition_fixes_20192.xls', sheet_name="pe_gmes_prod")
        # df_gmes_test = pd.read_excel('C:\\Users\\A0710996\\Desktop\\SWISSSTAFF_SSE_repartition_fixes_20192.xls', sheet_name="pe_gmes_qc_test")
        df_este_gestion = pd.read_excel('C:\\Users\\A0710996\\Desktop\\SWISSSTAFF_SSE_repartition_fixes_20192.xls', sheet_name="2015")
        
        df_este_gestion['No MUPE'] = df_este_gestion['No MUPE'].astype(int)
        df = pd.merge(df_este_gestion, df_gmes, how='outer', left_on=['No MUPE'], right_on=['NOM_ESTE'])
    
        # absents des gmes
        df[df['NOM_ESTE'].isnull()]

        # n'existe pas dans la liste de gestion car probablement entreprises fermée
        df_to_exclude = df[df['No MUPE'].isnull()]
       
   
    def generator(table, usr_id, no_ip, df_to_exclude):
        def generate_user_access(row):
            if df_to_exclude['CODE'].where(lambda x : x == row['No MUPE']).count() == 1:
                return None

            select = "(select c.id from companies c where c.code = '" + str(row['No MUPE']) + "' and c.cli_id = 1)"

            sql_insert : str = "insert into user_access (cli_id, com_id, usr_id, active) values ("
            sql_insert += '1,'
            sql_insert += select
            sql_insert += ',' 
            sql_insert += str(usr_id)
            sql_insert += ',' 
            sql_insert += "'" 
            sql_insert += 'Y'
            sql_insert += "'" 
            sql_insert += ');'
            return sql_insert

        def generate_companies(row):
            if df_to_exclude['CODE'].where(lambda x : x == row['No MUPE']).count() == 1:
                return None

            sql_insert : str = "insert into companies (cli_id, id, code, name, intmail, active) values ("
            sql_insert += '1,'
            # sans effet car il y aune séquence auto sur com_id
            sql_insert += str(row['ID'])
            sql_insert += ',' 
            sql_insert += str(row['No MUPE'])
            sql_insert += ',' 
            sql_insert += "'" 
            sql_insert += row["Nom de exact de l'entreprise"]
            sql_insert += "'" 
            sql_insert += ',' 
            sql_insert += "'" 
            sql_insert += 'pasquale.ferrara@aonhewitt.com'
            sql_insert += "'" 
            sql_insert += ',' 
            sql_insert += "'" 
            sql_insert += 'Y'
            sql_insert += "'" 
            sql_insert += ');'
            return sql_insert

        def generate_pe_gmes(row):
            pe_grmu_id = None

            # if df_to_exclude['CODE'].where(lambda x : x == row['No MUPE']).count() == 1:
            #     return None

            if row['No MUPE'] is None:
                return None

            collectif_norm = row['Collectifs_norm']
            if collectif_norm == 'C11':
                pe_grmu_id = 70
            elif collectif_norm == 'C21':
                pe_grmu_id = 75
            elif collectif_norm == 'C31':
                pe_grmu_id = 76
            elif collectif_norm == 'C41':
                pe_grmu_id = 77
            
            select = "(select pe_este_id from pe_este e where e.nom_este=" + str(row['No MUPE']) + " and e.no_ip = " + str(no_ip) + " and NO_NSTE=1)"
                
            if 0:
                sql_insert : str = "insert into pe_gmes (user_cre, dh_cre, pe_grmu_id, pe_este_id) values ("
                sql_insert += "'" 
                sql_insert += 'WRI'
                sql_insert += "'" 
                sql_insert += ',' 
                sql_insert += 'current_timestamp'
                sql_insert += ',' 
                sql_insert += str(pe_grmu_id)
                sql_insert += ',' 
                sql_insert += select
                sql_insert += ');'
                return sql_insert

            sql_update : str = "update pe_gmes"
            sql_update += " set user_MAJ = 'WRI'"
            sql_update += ",dh_maj = "
            sql_update += 'current_timestamp'
            sql_update += ",pe_grmu_id = "
            sql_update += str(pe_grmu_id)
            sql_update += " where pe_este_id = "
            sql_update += select
            sql_update += ';'
          
            return sql_update


        if table == 'companies':
            return generate_companies
        elif table == 'user_access':
            return generate_user_access
        elif table == 'pe_gmes':
            return generate_pe_gmes

    if 0:
        # f_generator = generator('companies', 0, 3490, df_to_exclude)
        # f_generator = generator('user_access', 657, 3490, df_to_exclude)
        f_generator = generator('pe_gmes', 657, 3490, df_to_exclude)
        # df_este_gestion.apply(f_generator, axis=1).to_csv('generate.csv', index=False, sep = '\t')
        df.apply(f_generator, axis=1).to_csv('generate.csv', index=False, sep = '\t')


except ValueError as e  :
    print (e)



       
