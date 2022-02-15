# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:54:14 2022

@author: adolescent
"""

import numpy as np


def Get_Average_Tuning(cell_lists,tuning_dic,target_tuning = 'LE',mode = 'Cohen_D'):
    '''
    Calculated averaged tuning of selected cells

    Parameters
    ----------
    cell_lists : (list)
        List of cells you want to average.
    tuning_dic : (dic)
        Dic of all cell tuning.
    target_tuning : (str), optional
        Tuning you want to average. Must be included in frame. The default is 'LE'.
    mode : ('Cohen_D','t_value',or'Tuning_Index'), optional
        Parameter you want to average. The default is 'Cohen_D'.

    Returns
    -------
    averaged_tuning : TYPE
        DESCRIPTION.

    '''
    all_tunings = np.zeros(len(cell_lists))
    for i,cc in enumerate(cell_lists):
        c_tuning = tuning_dic[cc][target_tuning]
        if (mode == 'Cohen_D') or (mode == 't_value') or (mode == 'Tuning_Index'):
            all_tunings[i] = c_tuning[mode]
        else:
            raise IOError('Invalid tuning mode.')
    # Average all tunings to get avr tuning.
    averaged_tuning = all_tunings.mean()
    return averaged_tuning