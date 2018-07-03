import os
from pathlib import PureWindowsPath
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import re as regexp

try :

    clientRootDir = PureWindowsPath('O:\\')
    workingDir = PureWindowsPath(os.getcwd())
    
    clientDir = PureWindowsPath(clientRootDir.joinpath('Lausanne','24 BenAdmin-Param','01 Analyse règlement','RGL_2018','Implementation','RSUPP'))
    dataFileOrigine = PureWindowsPath(clientDir.joinpath('data_rsupxx_QC.csv'))
    
    data = pd.read_csv(dataFileOrigine, sep = ';', encoding = "ascii")
    data.set_index(ibanData.NPERSO, inplace = True)
   

except ValueError as e  :
    print ("ERROR")
       