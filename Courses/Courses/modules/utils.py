import os
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class analyze:
    #------------------------------------------------------------------------------
    # analyze columns 
    #
    # applies series of threshes on the df and mesures the number
    # of remaining columns
    # 
    # plot the result  
    #------------------------------------------------------------------------------
    def analyze_columns(self, df, results):
        _df = df.copy()

        null_values_max = _df.isnull().sum().max()
        if null_values_max > 0:
            threshes = np.arange(0, null_values_max, 100)
            
            def select_columns(df, thresh):
                row = {}
                _df = df.dropna(thresh=thresh, axis=1)
                row['thresh'] = thresh
                row['shape'] = _df.shape[1]
                results.append(row)
        
            for thresh in threshes :
                select_columns(_df, thresh)
    
            df_results = pd.DataFrame(results)
            df_results.set_index('shape', inplace=True)
        
            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.suptitle("numbers of remaining columns in terms of thresh for a df of shape (" +  
                          str(df.shape[0]) + "," + str(df.shape[1]) + ")")
        
            x_pos = np.arange(len(df_results))
            ax.bar(x_pos, df_results.index, align='center', color='green')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_results['thresh'], rotation=45)

            ax.set_ylabel("remaining columns")
            ax.set_xlabel("thresh")

            plt.show()


class meta_data:
    #------------------------------------------------------------------------------
    # meta data 
    #
    #------------------------------------------------------------------------------

    # _df_meta : pd.DataFrame
    _working_dir : str

    def _init_(self, working_dir):
        _working_dir = working_dir


    def load_meta_data(self):
        data_file = os.path.join(_working_dir, 'course_projects', 'Data', 'module_3', 'meta_data.txt')
        _df_meta = pd.DataFrame.from_csv(data_file, sep='\t')
        _df_meta.reset_index(inplace=True)
        _df_meta.ffill(inplace=True)

