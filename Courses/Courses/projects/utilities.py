
import numpy as np
import pandas as pd

class utilities:
    #-------------------------------------------------------------------------
    # retourne la liste des noms de colonne d'un type donné
    #-------------------------------------------------------------------------
    def select_column_label(df, type):
        columns = df.dtypes[df.dtypes.apply(lambda x : x == type)].index.tolist()
        return columns

    #-------------------------------------------------------------------------
    # identifier et gérer les valeurs extrêmes 
    # on considère le seuil de 1% pour déterminer les valeurs extrêmes; ce seuil
    # determine (sous l'hypothèse de normalité de la distribution) toutes les valeurs 
    # x | x > abs(mean - 3 * std) 
    #-------------------------------------------------------------------------
    def remove_outliers(df, col_name):

        def reject_(ser : pd.Series):
            _mean = ser.mean() 
            _std_1  = 3 * ser.std()
  
            def f_(x):
                _reject = False

                if np.abs(x - _mean) > _std_1 : 
                    _reject = True
                return _reject
    
            return f_

        ser_col = df.loc[:,col_name]
        f_reject = reject_(ser_col)
        df = df.loc[~ser_col.apply(f_reject)]

        return df
 
   

