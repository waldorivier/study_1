import os
from pathlib import PureWindowsPath
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import re as regexp

try :

    clientRootDir = PureWindowsPath('O:\\')
    workingDir = PureWindowsPath(os.getcwd())

    clientDir = PureWindowsPath(clientRootDir.joinpath('CAP','24 BenAdmin-Param','05 Paiement','SEPA'))
    dataFileOrigine = PureWindowsPath(clientDir.joinpath('5_adresse_paiement_correction_is_iban.xlsx'))
    dataFileCsvOrigine = PureWindowsPath(clientDir.joinpath('20_adresse_paiement_correction_Autres.csv'))
    
    ibanData = pd.read_excel(dataFileOrigine.as_posix())

    ibanData.set_index(ibanData.NPERSO, inplace = True)
    ibanData.STYPAIE = ibanData.STYPAIE.map({7:'IBAN/International'})

    prog = regexp.compile('^CH')


except ValueError as e  :
    print ("ERROR")
       