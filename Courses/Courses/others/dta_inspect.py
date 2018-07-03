import os
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
import xml.etree.cElementTree as et
from xml.dom import minidom

try :
    data_file_name = 'dta_xml_rente_juin_5750_18_093_qfdo42441.xml'

    client_root_dir = PureWindowsPath('O:\\')
    working_dir = PureWindowsPath(os.getcwd())
    data_dir = PureWindowsPath(client_root_dir.joinpath('Lausanne','24 BenAdmin-Param','05 Paiement','SEPA','Contrôles'))
    data_file = PureWindowsPath(data_dir.joinpath(data_file_name))

    doc = et.parse(data_file.as_posix())
    
    # doc = minidom.parse(data_file.as_posix())
    
    doc_root = doc.getroot()
    for node in doc_root.findall('.//CdtTrfTxInf'):
        print (node)

    #   email = node.find('email')
    #   phone = node.find('phone')
    #    street = node.find('address/street')
    #    df_xml = df_xml.append(
    #        pd.Series([name, getvalueofnode(email), getvalueofnode(phone),
    #                   getvalueofnode(street)], index=dfcols),
    #        ignore_index=True)
 
    #print df_xml
    
    

except ValueError as e  :
    print ("ERROR")

