# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 14:04:56 2021

@author: ZR
"""


import numpy as np

def Most_Correlated_Index(correlation_plot,mode = 'High',prop = 0.1):
    
    '''
    Get ID list of most/least correlated frames 

    Parameters
    ----------
    correlation_plot : (array)
        Correlation plot. Need to be 1D array.
    mode : ('High' or 'Low'), optional
        Mode of correlation find. Highest or Lowest . The default is 'High'.
    prop : (float), optional
        Propotion of correlated graphs. The default is 0.1.

    Returns
    -------
    ID_lists : (array-list)
        List of correlation ID in condition above.

    '''    
    
    frame_num = int(len(correlation_plot)*prop)
    if mode == 'High':
        ID_list = np.argpartition(correlation_plot,-frame_num)[-frame_num:]
    elif mode == 'Low':
        ID_list = np.argpartition(correlation_plot,frame_num)[:frame_num]
    else:
        raise IOError('Invalid sorting method.')
        
    return ID_list