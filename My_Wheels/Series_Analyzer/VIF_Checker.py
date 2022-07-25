# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:36:40 2022

@author: ZR
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np


def VIF_Check(data_frame,checked_column_name):
    '''
    Calculate vif, near 1 is independent, higher indicate correlated.

    Parameters
    ----------
    data_frame : (pd Frame)
        Data frame of correlation, each index is a piece of data.
    checked_column_name : (list)
        List of columns .

    Returns
    -------
    vif_index : TYPE
        DESCRIPTION.

    '''
    reformed_data = data_frame[checked_column_name]
    reformed_data['c'] = 1
    name = reformed_data.columns
    x = np.matrix(reformed_data)
    VIF_list = [variance_inflation_factor(x,i) for i in range(x.shape[1])]
    VIF = pd.DataFrame({'feature':name,"VIF":VIF_list})
    
    return VIF